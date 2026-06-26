# Lesson 00: 项目地图与一次推理请求的生命线

## 本课目标

学完这一课，你要能从整体上回答：

- nano-vllm 为什么适合作为 vLLM 入门项目？
- 一次 `LLM.generate()` 请求经过哪些对象？
- `Sequence`、`Scheduler`、`ModelRunner` 分别负责什么？
- prefill 和 decode 为什么是两个不同阶段？

## 当前学习状态

本课主体内容已经覆盖：

- LLM 推理大图。
- prompt 到 token id 的转换。
- `Sequence` 为什么代表请求运行时状态。
- block table 的第一层直觉。
- `LLMEngine.generate()`、`add_request()`、`step()` 的主路径。

本课还需要一次复述确认，才算完全收尾。确认问题见文末“检查点”。

Lesson 01 并不是跳过 Lesson 00，而是从本课主路径中的 `scheduler.schedule()` 自然展开，专门学习调度器。

## 背景：为什么第一课先看项目地图

大模型推理系统很容易让新手迷路，因为它同时出现了很多层概念：tokenizer、Transformer、KV cache、scheduler、GPU、显存、batch、采样。第一课不要求你掌握所有细节，只要求你先知道“一次请求在系统里怎么流动”。

这相当于先拿到城市地图：

- 你不需要马上知道每条路怎么修。
- 但你要知道入口在哪里，主干道怎么走，哪些建筑以后会反复进去。

nano-vllm 的好处是代码量小，但组件完整。我们可以用它建立 vLLM 的核心地图，再逐步补 Transformer、KV cache、PagedAttention、GPU 执行等背景。

## 本课的学习粒度

这一课先刻意不深入：

- 不推导 attention 公式。
- 不讲 CUDA kernel 细节。
- 不分析 FlashAttention。
- 不追 tensor parallel 的通信。

我们只追一件事：一个 prompt 进入 `generate()` 后，如何变成 `Sequence`，如何被 scheduler 调度，如何通过 model runner 生成 token，最后如何回到文本。

## 本课教学顺序

这一课不能一上来就读源码。正确顺序是：

1. 先讲背景：为什么推理系统需要把一次请求拆成多个阶段。
2. 再讲直觉：一次生成请求像一张“工单”，不是一段静态字符串。
3. 然后看源码：`LLMEngine` 和 `Sequence` 如何表达这张工单的生命周期。
4. 最后回扣：源码里的 `generate()`、`add_request()`、`step()` 分别对应推理大图里的哪一步。

之后每进入一个新模块，也按这个顺序走：先问“为什么需要它”，再问“它怎么做”，最后才看“代码在哪里”。

## 项目地图

核心文件可以先这样分组：

- 用户入口：`nanovllm/llm.py`、`example.py`
- 引擎主循环：`nanovllm/engine/llm_engine.py`
- 请求状态：`nanovllm/engine/sequence.py`
- 调度：`nanovllm/engine/scheduler.py`
- KV cache 分块管理：`nanovllm/engine/block_manager.py`
- GPU/model 执行：`nanovllm/engine/model_runner.py`
- 模型结构：`nanovllm/models/qwen3.py`
- 神经网络层：`nanovllm/layers/`
- 全局 attention 上下文：`nanovllm/utils/context.py`

## 一条请求的主路径

1. 用户创建 `LLM(model_path, ...)`。
2. `LLM` 继承 `LLMEngine`，真正逻辑在 `LLMEngine.__init__`。
3. `generate(prompts, sampling_params)` 把每个 prompt 加入 scheduler。
4. `add_request()` 把字符串 prompt tokenize 成 token id，然后包装成 `Sequence`。
5. `step()` 调用 `scheduler.schedule()`，决定这轮跑 prefill 还是 decode。
6. `model_runner.call("run", seqs, is_prefill)` 准备张量，执行模型，采样下一个 token。
7. `scheduler.postprocess()` 更新缓存状态、追加 token、判断是否结束。
8. 所有 sequence 完成后，token ids 被 decode 回文本。

## 源码主循环：先抓住三段

第一次读 `LLMEngine`，不要把注意力放在所有细节上。先抓住三段：

### 1. 初始化阶段：准备长期存在的组件

对应 `nanovllm/engine/llm_engine.py` 的 `LLMEngine.__init__()`。

它做的事情可以理解成“开店前准备工具”：

- 创建 `Config`：保存模型路径、最大 batch token 数、KV cache block size、tensor parallel 大小等配置。
- 设置 `Sequence.block_size`：让所有请求用同一个 KV cache block 大小。
- 创建 `ModelRunner`：负责真正执行模型 forward。
- 创建 tokenizer：负责字符串和 token id 的互相转换。
- 创建 `Scheduler`：负责管理等待中和运行中的请求。

这一步不是在生成文本，而是在搭建推理引擎的运行环境。

### 2. 加入请求阶段：外部输入变成内部状态

对应 `LLMEngine.add_request()`：

```python
if isinstance(prompt, str):
    prompt = self.tokenizer.encode(prompt)
seq = Sequence(prompt, sampling_params)
self.scheduler.add(seq)
```

这里发生了一个非常重要的转换：

```text
用户字符串 prompt -> token ids -> Sequence -> scheduler waiting queue
```

为什么要这样转？

- 模型只能处理 token id，不能直接处理字符串。
- scheduler 调度的是请求状态，不是原始文本。
- 生成过程中还要不断追加 token、更新缓存、判断结束，所以必须有 `Sequence`。

### 3. 单步推进阶段：每次让系统往前走一步

对应 `LLMEngine.step()`：

```python
seqs, is_prefill = self.scheduler.schedule()
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

这三行是推理系统的核心节奏：

1. scheduler 决定这一轮跑哪些请求，以及是 prefill 还是 decode。
2. model runner 执行模型，采样出下一个 token。
3. scheduler 把新 token 写回 sequence，更新状态，必要时释放资源。

你可以把 `step()` 理解成推理引擎的“心跳”。每跳一次，系统要么处理一批 prompt token，要么为一批请求各生成一个 token。

## `generate()` 为什么是一个 while 循环

`generate()` 不是调用模型一次就结束，而是：

```text
把所有 prompt 加入 scheduler
while 还有请求没完成:
    step()
整理输出
```

原因是自回归生成天然就是循环：模型每次通常只决定下一个 token，生成出来以后还要追加回上下文，再继续下一步。直到遇到 EOS，或者达到 `max_tokens`。

所以你在推理系统里经常会看到两层循环：

- 外层：服务持续接收请求。
- 内层：每个请求持续生成 token，直到完成。

nano-vllm 的离线 `generate()` 展示的是内层循环。

## 关键概念

### Sequence

`Sequence` 是一次生成请求在系统内部的状态对象。它不只是 prompt，还保存：

- 当前 token 列表
- prompt token 数
- completion token 数
- 已缓存 token 数
- 本轮要调度 token 数
- block table
- 采样参数
- 是否完成

为什么不能只传字符串？因为推理过程中，请求会不断变化。字符串只表示最初的输入，而系统还需要知道：

- 已经生成了多少 token。
- 哪些 token 已经写入 KV cache。
- 这一轮该跑 prefill 还是 decode。
- 这条请求占用了哪些 KV cache block。
- 它是否已经遇到 EOS 或达到最大生成长度。

所以 `Sequence` 更像“请求运行时状态”，不是普通输入文本。

### Prefill

Prefill 是“处理 prompt”的阶段。输入可能是一大段 prompt token，模型一次性计算这些 token 的 hidden states，同时把每层 attention 需要复用的 K/V 写入 KV cache。

直觉：读完题目，建立上下文。

### Decode

Decode 是“逐 token 生成答案”的阶段。每轮通常只给模型新生成的最后一个 token，但 attention 会通过 KV cache 看到之前的上下文。

直觉：每次写下一个字，同时翻看已经读过/写过的上下文缓存。

### Block Table

block table 是 sequence 到 KV cache 物理块的映射。它让每个请求不需要占用一整段连续显存，而是像操作系统分页一样使用固定大小 block。

在 nano-vllm 中，相关逻辑集中在：

- `Sequence.block_table`
- `BlockManager.allocate()`
- `BlockManager.can_append()`
- `ModelRunner.prepare_prefill()`
- `ModelRunner.prepare_decode()`

常见误区：

- block table 不是 token 内容列表。
- block table 记录的是逻辑 token 块到 KV cache 物理块的映射。
- 一个 sequence 的 token 内容在 `token_ids` 中；它的缓存位置在 `block_table` 中。

## 本课源码阅读顺序

1. `nanovllm/llm.py`
2. `nanovllm/engine/llm_engine.py`
3. `nanovllm/engine/sequence.py`
4. `nanovllm/engine/scheduler.py` 的 `schedule()`

先不要急着看 attention 和 CUDA。第一遍只追“请求状态如何变化”。

## 练习

运行：

```bash
python3 learning/exercises/lesson_00_sequence_blocks.py
```

观察输出里的：

- prompt tokens
- block table
- prefill 后状态
- decode 追加 token
- completion tokens

然后尝试修改练习里的 `block_size`、`prompt`、`generated_tokens`，观察 block table 什么时候增加新块。

## 检查点

请用自己的话解释：

1. 为什么推理系统内部要有 `Sequence`，而不是直接传字符串？
2. prefill 和 decode 的输入长度有什么不同？
3. block table 记录的是 token 内容，还是 token 在 KV cache 中的位置？
4. 为什么 block size 会影响显存管理？
