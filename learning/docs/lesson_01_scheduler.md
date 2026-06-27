# Lesson 01: Scheduler 背景、队列和推理调度

## 本课目标

学完这一课，你要能解释：

- 为什么大模型推理系统需要 scheduler。
- 单请求生成和多请求生成有什么不同。
- waiting 队列和 running 队列分别代表什么。
- prefill 为什么通常优先于 decode。
- decode 为什么适合把多个请求合成 batch。
- `max_num_seqs` 和 `max_num_batched_tokens` 控制的是什么资源。

## 背景问题：为什么需要调度器

如果只有一个用户、一个 prompt，推理流程很简单：

```text
prefill prompt -> decode token 1 -> decode token 2 -> ... -> finish
```

但真实推理服务通常同时面对很多请求：

- 请求 A 的 prompt 很短，只想生成 20 个 token。
- 请求 B 的 prompt 很长，要先处理几千个 prompt token。
- 请求 C 已经完成 prefill，正在每轮生成一个 token。
- 请求 D 刚刚到来，还没进入 GPU。

如果没有 scheduler，系统会遇到几个问题：

1. GPU 可能吃不饱：一次只跑一个请求，很多算力浪费。
2. 长 prompt 可能阻塞短请求：一个超长 prefill 把后面的请求都堵住。
3. 显存可能爆掉：每个请求都需要 KV cache，不能无脑全放进去。
4. 请求结束后资源需要回收：否则显存会越来越少。

所以 scheduler 的职责不是“让模型更聪明”，而是“决定每一轮让哪些请求占用 GPU 和 KV cache”。

## 核心直觉：餐厅排队和厨房出餐

可以把推理系统想象成餐厅：

- 用户请求：一张订单。
- prompt prefill：厨房先读完整个订单，准备上下文。
- decode：每轮出一道小菜，也就是生成一个 token。
- GPU：厨房灶台，能同时处理一批菜，但容量有限。
- KV cache：已经准备好的半成品和上下文，占用冰箱/台面空间。
- scheduler：前台加厨房调度员。

调度员需要决定：

- 哪些新订单可以进厨房。
- 哪些已开工订单这轮继续出菜。
- 如果空间不够，是否先暂停某些订单。
- 已完成订单释放占用的台面。

这就是 waiting/running 队列的直觉。

## 两个队列

### waiting

waiting 表示“请求已经进入系统，但还没有完成 prefill”。

这些请求通常还没建立完整 KV cache。它们需要先处理 prompt，才能进入逐 token 生成阶段。

### running

running 表示“请求已经完成 prefill，正在 decode”。

这些请求每一轮通常只需要输入上一个 token，然后生成下一个 token。它们已经有历史上下文的 KV cache。

## Prefill 和 Decode 的调度差异

### Prefill

prefill 一次可能处理很多 prompt token。它的特点是：

- token 数可能很大。
- 需要写入大量 KV cache。
- 适合按 token 总数控制 batch。

所以 scheduler 会看 `max_num_batched_tokens`：这一轮最多处理多少 prompt token。

### Decode

decode 阶段每个请求通常只生成一个 token。它的特点是：

- 单个请求每轮 token 数很小。
- 多个请求可以合在一起跑，让 GPU 更忙。
- 适合按请求数控制 batch。

所以 scheduler 会看 `max_num_seqs`：这一轮最多让多少条 sequence 一起跑。

## 为什么 prefill 优先

nano-vllm 的 `schedule()` 先尝试调度 waiting 队列里的 prefill。如果有 prefill 可跑，就直接返回 prefill batch；只有没有 prefill 可跑时，才进入 decode。

这种策略的直觉是：

- 新请求必须先完成 prefill，才能进入 running。
- prefill 完成后，这个请求才能参与后续 decode batch。
- 如果一直只 decode 老请求，新请求会一直进不来。

但这也带来取舍：prefill 太大时可能影响已有 running 请求的 decode 延迟。成熟推理系统会围绕这个点做更复杂的调度策略。

## nano-vllm 源码锚点

对应文件：`nanovllm/engine/scheduler.py`

核心成员：

- `self.waiting`：等待 prefill 的请求队列。
- `self.running`：已经进入 decode 的请求队列。
- `self.block_manager`：负责 KV cache block 分配和释放。
- `self.max_num_seqs`：一轮最多调度多少条 sequence。
- `self.max_num_batched_tokens`：一轮 prefill 最多处理多少 token。

核心方法：

- `add(seq)`：新请求进入 waiting。
- `is_finished()`：waiting 和 running 都空，说明全部结束。
- `schedule()`：决定本轮跑 prefill 还是 decode。
- `postprocess()`：模型执行后，更新缓存、追加 token、判断完成。
- `preempt(seq)`：资源不足时，把 running 请求退回 waiting。

## 用源码回扣背景

从背景看，scheduler 需要回答三个问题：

1. 新请求在哪里等？
2. 正在生成的请求在哪里继续跑？
3. 每一轮 GPU 到底跑哪一批？

nano-vllm 的回答是：

- 新请求进 `waiting`。
- prefill 完成后移到 `running`。
- `schedule()` 先尝试从 `waiting` 组成 prefill batch。
- 如果没有 prefill batch，再从 `running` 组成 decode batch。
- 每轮调度后，`postprocess()` 更新状态，完成的请求释放 block。

## 常见误区

- scheduler 不负责模型数学计算，模型 forward 在 `ModelRunner`。
- scheduler 不直接存 KV tensor，它通过 `BlockManager` 管理 block。
- waiting 不是“还没进系统”，而是“进系统了，但还没完成 prefill”。
- running 不是“正在占用 CPU 线程”，而是“已经完成 prefill，处于可 decode 状态”。
- decode batch 的 token 数通常等于请求数，因为每条请求每轮生成一个 token。

## 练习

运行：

```bash
python3 learning/exercises/lesson_01_toy_scheduler.py
```

观察：

- 请求什么时候从 waiting 进入 running。
- 有 waiting 时为什么优先 prefill。
- 没有 waiting 后，decode 如何一次调度多个 running 请求。
- 请求达到 `max_new_tokens` 后如何完成并离开 running。

## 检查点

请用自己的话解释：

1. scheduler 解决的是模型能力问题，还是系统资源调度问题？
2. waiting 和 running 的区别是什么？
3. prefill batch 为什么更关注 token 总数？
4. decode batch 为什么更关注请求数量？
5. 为什么完成的请求必须释放 KV cache block？
## 2026-06-27 课堂复盘

本节从“很多请求同时来，GPU/KV cache/token batch 资源有限”这个背景进入 scheduler，而不是直接读源码。

学习者已经运行：

```bash
python3 learning/exercises/lesson_01_toy_scheduler.py
```

输出中的关键现象：

```text
waiting：还没完成 prefill 的请求
running：prefill 已完成，已经拥有可复用的 KV cache，可以参与 decode 的请求
prefill batch：本轮在处理 prompt，可能一次处理多个 token
decode batch：本轮给多个请求各生成 1 个 token
```

特别要保留的易混点：

```text
prefill batch 里出现某个请求，不代表它已经完成 prefill。
如果 prompt 还没全部处理完，它仍然留在 waiting。
```

学习者已经能解释：

- A 还没开始时，如果资源够就 prefill，资源不够就排队。
- B 已经 prefill 完并正在生成时，可以进入 decode。
- C 已经结束时，需要释放 KV cache。
- toy 输出 step 2 中 B 出现在 `prefill batch`，但仍在 `waiting`，是因为 B 还没有 prefill 完成。

下次继续本课时，先复述 scheduler 的一句话定义：

```text
Scheduler 在有限 GPU/KV cache/token batch 资源下，决定每一轮让哪些请求运行。
```

然后再进入源码：

- `nanovllm/engine/sequence.py`：一个请求如何记录 prompt、生成 token、完成状态。
- `nanovllm/engine/scheduler.py`：waiting/running 队列和资源限制。
- `nanovllm/engine/llm_engine.py`：每轮 `step()` 如何调度并调用模型执行。
