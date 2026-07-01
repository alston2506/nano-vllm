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
- KV cache 不是只缓存 prefill token。prefill 会把 prompt token 的 KV 写入缓存；decode 每轮处理上一个 token 时，也会把这个被处理 token 的 KV 写入缓存。本轮刚采样出来的新 token 会追加到 `Sequence`，通常在下一轮 decode 被模型处理后才拥有自己的 KV cache。

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

另一个易混点：`max_num_batched_tokens` 还有剩余时，为什么有时不会继续塞下一个请求。

toy scheduler 和 nano-vllm 源码都采用一个简化规则：

```text
允许一个 prefill batch 里有多个请求；
但如果某个请求放进来时必须被切成一小段，它只能作为本轮 batch 的第一条请求。
```

对应 toy 代码：

```python
if remaining_prompt > remaining_budget and scheduled:
    break
```

读法是：

```text
如果当前请求剩余 prompt token 数 > 本轮剩余 token budget，
并且本轮已经安排过别的请求，
就不要把这个请求切开塞进当前 batch。
```

所以 `max_num_batched_tokens=6` 时：

```text
step 1: A 需要 4 个 token，可以完整放入；剩余 budget=2。
        B 还剩 8 个 token，放不完整，而且 A 已经在 batch 里，所以不放 B。

step 2: B 是本轮第一条请求，允许被切开，于是处理 B 的 6 个 token。

step 3: B 只剩 2 个 token，可以完整放入；剩余 budget=4。
        C 需要 2 个 token，也可以完整放入，所以本轮 batch 是 [B, C]。
```

学习者已经能解释：

- A 还没开始时，如果资源够就 prefill，资源不够就排队。
- B 已经 prefill 完并正在生成时，可以进入 decode。
- C 已经结束时，需要释放 KV cache。
- toy 输出 step 2 中 B 出现在 `prefill batch`，但仍在 `waiting`，是因为 B 还没有 prefill 完成。

toy scheduler 里没有真实 KV cache block，所以没有显式的 `deallocate()`。
它用一个简化动作表达“请求完成并释放资源”：

```python
request.generated_tokens += 1
if not request.finished:
    self.running.append(request)
```

如果请求已经 `finished`，它就不会被重新放回 `running` 队列。
这表示 toy 系统里该请求已经离开运行集合；真实 nano-vllm 中对应的是 `Scheduler.postprocess()` 里调用 `block_manager.deallocate(seq)` 释放 KV cache block。

下次继续本课时，先复述 scheduler 的一句话定义：

```text
Scheduler 在有限 GPU/KV cache/token batch 资源下，决定每一轮让哪些请求运行。
```

然后再进入源码：

- `nanovllm/engine/sequence.py`：一个请求如何记录 prompt、生成 token、完成状态。
- `nanovllm/engine/scheduler.py`：waiting/running 队列和资源限制。
- `nanovllm/engine/llm_engine.py`：每轮 `step()` 如何调度并调用模型执行。

## 2026-06-28 源码锚点：Sequence -> Scheduler -> LLMEngine.step

今天从 toy scheduler 过渡到 nano-vllm 源码。先记住一条最短路径：

```text
LLMEngine.add_request()
-> Sequence
-> Scheduler.add(seq)
-> Scheduler.schedule()
-> ModelRunner.run(...)
-> Scheduler.postprocess(...)
```

### Sequence：一条请求的账本

对应文件：`nanovllm/engine/sequence.py`

`Sequence` 不做模型计算，它记录一条请求当前走到哪里：

- `status`：`WAITING`、`RUNNING`、`FINISHED`。
- `token_ids`：prompt token 加上后续生成的 completion token。
- `num_prompt_tokens`：prompt 和 completion 的分界线。
- `num_tokens`：当前这条 sequence 的总 token 数，开始等于 prompt 长度，后续每生成一个 token 就加 1。
- `num_cached_tokens`：已经进入 KV cache 的 token 数。
- `num_scheduled_tokens`：本轮 scheduler 安排模型处理的 token 数。
- `is_prefill`：这条请求本轮是否处于 prefill 路径。
- `block_table`：这条请求占用的 KV cache block id 列表。

`num_tokens` 和 `num_prompt_tokens` 初始化时相同，但不是同一个含义：

```text
num_prompt_tokens 固定表示原始 prompt 有多长。
num_tokens 表示当前总长度，会随着 decode 生成新 token 增长。
```

例如 prompt 有 5 个 token，之后生成了 3 个 token：

```text
num_prompt_tokens = 5
num_tokens = 8
num_completion_tokens = num_tokens - num_prompt_tokens = 3
```

初始化代码里：

```python
self.token_ids = copy(token_ids)
self.num_tokens = len(self.token_ids)
self.num_prompt_tokens = len(token_ids)
```

`num_tokens` 用 `self.token_ids`，表示它跟内部维护的当前 token 列表绑定；后续 `append_token()` 会更新这份内部列表和 `num_tokens`。

`num_prompt_tokens` 用传入的 `token_ids`，表示它记录的是原始 prompt 的长度。初始化那一刻，`len(token_ids)` 和 `len(self.token_ids)` 一样；这里更多是表达语义差异，而不是功能上必须这样写。

直觉：

```text
Sequence 像一张订单的进度表。
Scheduler 每轮看这些进度表，决定哪些订单能继续推进。
```

### Scheduler.schedule：先 prefill，后 decode

对应文件：`nanovllm/engine/scheduler.py`

`schedule()` 的主结构是：

```text
先尝试从 waiting 组成 prefill batch。
如果 prefill batch 不为空，直接返回。
如果没有 prefill 可跑，再从 running 组成 decode batch。
```

prefill 阶段关注：

- `max_num_batched_tokens`：本轮最多处理多少 prompt token。
- KV cache block 是否能分配。
- prompt 是否已经全部缓存完。

decode 阶段关注：

- `max_num_seqs`：本轮最多让多少条 sequence 一起 decode。
- 每条 running sequence 本轮通常只安排 1 个 token。
- 追加新 token 的 KV cache block 是否足够。

### postprocess：模型跑完之后收账

`Scheduler.postprocess()` 做三件事：

1. 更新 block hash 和 `num_cached_tokens`。
2. 如果本轮已经能产出 next token，就 `append_token(token_id)`。
3. 如果遇到 eos 或达到 `max_tokens`，标记 `FINISHED`，释放 KV cache block，并从 `running` 移除。

toy scheduler 里没有 `BlockManager`，所以它用“不再 append 回 running”表示请求结束。真实 nano-vllm 里会显式调用：

```python
self.block_manager.deallocate(seq)
self.running.remove(seq)
```

### LLMEngine.step：一轮推理心跳

对应文件：`nanovllm/engine/llm_engine.py`

`step()` 可以读成：

```text
问 scheduler：本轮谁跑？
让 model_runner：真的跑模型。
让 scheduler：根据模型输出更新请求状态。
```

源码里的三行主干：

```python
seqs, is_prefill = self.scheduler.schedule()
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

这就是 nano-vllm 推理服务的心跳。

### Scheduler.schedule 逐段阅读

`schedule()` 的返回值是：

```python
tuple[list[Sequence], bool]
```

含义：

```text
第一个值：本轮要交给模型运行的 sequence 列表。
第二个值：本轮是不是 prefill。True 表示 prefill，False 表示 decode。
```

开头：

```python
scheduled_seqs = []
num_batched_tokens = 0
```

`scheduled_seqs` 暂存本轮选中的请求。`num_batched_tokens` 只在 prefill 阶段用来统计本轮已经安排了多少 prompt token。

补充概念：prefix cache 命中。

```text
prefix cache 指的是：如果新请求的 prompt 前缀和之前某个请求的 prompt 前缀相同，
并且那段前缀的 KV cache block 还可以复用，
系统就不必重新计算这部分 token。
```

例子：

```text
请求 1: "你是一个翻译助手。请翻译：hello"
请求 2: "你是一个翻译助手。请翻译：world"
```

两个请求开头的系统提示相同。对请求 2 来说，前面那段相同前缀如果已经在 KV cache block 中，就可能命中 prefix cache。

在 `schedule()` 中，`can_allocate(seq)` 返回的 `num_cached_blocks` 表示有多少个完整 block 的前缀可以复用。于是：

```python
num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
```

表示这条请求不需要从第一个 token 开始全部重新 prefill，只需要处理未命中的后半段。

prefill 循环：

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
```

只要 waiting 队列不空，并且本轮 sequence 数还没超过 `max_num_seqs`，就尝试继续从 waiting 里拿请求做 prefill。

```python
seq = self.waiting[0]
remaining = self.max_num_batched_tokens - num_batched_tokens
```

只看 waiting 队头请求。`remaining` 是本轮 prefill token budget 还剩多少。

```python
if remaining == 0:
    break
```

本轮 token budget 用完，就不能再安排 prefill。

```python
if not seq.block_table:
    num_cached_blocks = self.block_manager.can_allocate(seq)
    if num_cached_blocks == -1:
        break
    num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
else:
    num_tokens = seq.num_tokens - seq.num_cached_tokens
```

如果这个请求还没有分配过 KV cache block，就先问 `BlockManager` 能不能分配。`-1` 表示 block 不够，本轮 prefill 到此停止。否则计算这个请求还需要处理多少 token。

如果已经有 `block_table`，说明它之前做过 chunked prefill，只是 prompt 还没全部处理完。这时直接用 `seq.num_tokens - seq.num_cached_tokens` 计算剩余 prompt token。

```python
if remaining < num_tokens and scheduled_seqs:
    break
```

如果当前请求完整放不进剩余 token budget，并且本轮已经安排过别的请求，就不把它切开塞进当前 batch。换句话说，只允许本轮第一条请求被 chunked prefill。

```python
if not seq.block_table:
    self.block_manager.allocate(seq, num_cached_blocks)
```

如果还没分配 block，现在正式给它分配 KV cache block。

```python
seq.num_scheduled_tokens = min(num_tokens, remaining)
num_batched_tokens += seq.num_scheduled_tokens
```

决定这个请求本轮实际处理多少 token，并累计到本轮 token budget。

```python
if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
    seq.status = SequenceStatus.RUNNING
    self.waiting.popleft()
    self.running.append(seq)
```

如果本轮处理完后，prompt 的所有 token 都会进入 KV cache，这个请求就完成 prefill，从 `waiting` 移到 `running`。

```python
scheduled_seqs.append(seq)
```

把这个请求加入本轮要交给模型执行的列表。

```python
if scheduled_seqs:
    return scheduled_seqs, True
```

只要本轮安排到了 prefill，就直接返回，不再同一轮安排 decode。`True` 表示本轮是 prefill。

decode 循环：

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
```

只有没有 prefill 可跑时，才从 running 队列里选择请求做 decode。

```python
seq = self.running.popleft()
```

从 running 队头取出一条请求。

```python
while not self.block_manager.can_append(seq):
```

decode 会追加一个新 token，因此可能需要新的 KV cache block。这里检查能不能追加。

```python
if self.running:
    self.preempt(self.running.pop())
else:
    self.preempt(seq)
    break
```

如果 KV cache 不够，就抢占请求：优先把 running 队尾的请求退回 waiting；如果没有别的请求可抢占，就抢占当前请求。

```python
else:
    seq.num_scheduled_tokens = 1
    seq.is_prefill = False
    self.block_manager.may_append(seq)
    scheduled_seqs.append(seq)
```

如果 KV cache 可以追加，就把当前请求安排进 decode batch。decode 每条请求本轮只处理 1 个 token，所以 `num_scheduled_tokens = 1`。

```python
assert scheduled_seqs
self.running.extendleft(reversed(scheduled_seqs))
return scheduled_seqs, False
```

decode 至少应该安排到一条请求。安排好的请求放回 running 队列，等待模型输出后由 `postprocess()` 判断是否完成。`False` 表示本轮是 decode。

### Scheduler.postprocess 逐行阅读

对应文件：`nanovllm/engine/scheduler.py`

`schedule()` 负责决定“这一轮谁去跑模型”。`postprocess()` 负责在模型跑完后，更新这些请求的账本：

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

函数参数：

```text
seqs：这一轮刚刚被模型处理过的请求列表。
token_ids：模型给每个请求预测出的下一个 token。
is_prefill：这一轮是不是 prefill。
```

逐行理解：

```python
for seq, token_id in zip(seqs, token_ids):
```

把请求和模型输出一一配对。例如 `seqs = [A, B]`，`token_ids = [18, 42]`，表示 A 得到 token 18，B 得到 token 42。

```python
self.block_manager.hash_blocks(seq)
```

给已经写入 KV cache 的完整 block 计算 hash，并登记到 prefix cache 索引里。它不是在生成 token，而是在记录“这段 KV cache 以后可能被相同前缀复用”。

```python
seq.num_cached_tokens += seq.num_scheduled_tokens
```

模型已经处理完本轮安排的 token，所以这些 token 对应的 KV cache 已经存在。把本轮处理量计入 `num_cached_tokens`。

```python
seq.num_scheduled_tokens = 0
```

清空本轮临时调度量。下一轮 `schedule()` 会重新设置它。

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
```

如果当前是 prefill，并且 prompt 还没全部进入 KV cache，就先跳过后面的 `append_token()`。这是 chunked prefill 的关键：prompt 没处理完时，还没有真正进入 completion token 生成阶段。

```python
seq.append_token(token_id)
```

把模型输出的 token 加到这条 `Sequence` 里。`Sequence.append_token()` 会做三件事：

```python
self.token_ids.append(token_id)
self.last_token = token_id
self.num_tokens += 1
```

所以 `num_tokens` 会随着 decode 增长，而 `num_prompt_tokens` 固定记录原始 prompt 长度。

```python
if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
```

判断请求是否结束。结束条件有两个：

```text
1. 模型输出了 EOS，并且没有设置 ignore_eos。
2. 生成 token 数量已经达到 max_tokens。
```

```python
seq.status = SequenceStatus.FINISHED
```

把请求状态标记为完成。完成后不应该再被 scheduler 调度。

```python
self.block_manager.deallocate(seq)
```

释放这条请求占用的 KV cache block。KV cache 在 GPU 显存里，空间有限；请求结束后如果不释放，后面的请求就无法复用这些 block。

```python
self.running.remove(seq)
```

把完成的请求从 `running` 队列移除。

这一段可以压缩成一条链路：

```text
模型输出 token_id
-> 更新 KV cache 进度
-> 如果 prefill 还没完成，continue
-> 否则 append_token
-> 判断 EOS 或 max_tokens
-> FINISHED 后释放 KV cache block
-> 从 running 移除
```

学习者已能复述的关键点：

```text
prefill 未结束时，先不走 append_token 的逻辑。
append_token 之后，这个 Sequence 的 token 数量会 +1。
请求结束后必须释放 KV cache。
```

## 2026-07-01 BlockManager：KV cache block、prefix cache 和抢占

今天从 `Scheduler.postprocess()` 里的 `block_manager.deallocate(seq)` 接到 `BlockManager`。

核心背景：

```text
token_ids 是 CPU 侧的 token 账本。
KV cache 是模型跑完后写在 GPU 显存里的 K/V 张量。
block_table 是这条 Sequence 的逻辑 block 到物理 KV cache block 的映射。
```

例如 `block_size = 4`，prompt 有 6 个 token：

```text
tokens:      [10, 11, 12, 13, 14, 15]
logical:     block 0          block 1
block_table: [3,               8]
```

读法：

```text
逻辑 block 0 的 KV cache 写到物理 block 3。
逻辑 block 1 的 KV cache 写到物理 block 8。
```

### allocate 与 may_append

`allocate(seq, num_cached_blocks)` 用在 prefill 前。它为当前已有的一整段 prompt 或上下文准备 KV cache block。

```text
allocate = 先给这段上下文占好 KV cache 写入位置。
```

如果 prompt 长度为 6，`block_size = 4`，通常需要 2 个 block。`allocate()` 会把物理 block id 填进 `seq.block_table`。

`may_append(seq)` 用在 decode 前。decode 每轮通常只处理 1 个 token，所以它只关心“这轮要处理的最后一个 token 的 KV cache 是否需要新 block”。

关键条件：

```python
len(seq) % self.block_size == 1
```

读法：

```text
当前最后一个 token 是某个 block 的第一个 token。
如果是，就需要先追加一个新物理 block。
```

例子：`block_size = 4`，prompt 长度是 4。prefill 后模型生成第一个 completion token：

```text
token_ids = [10, 11, 12, 13, 14]
len(seq) = 5
5 % 4 == 1
```

token `14` 是新 block 的第一个 token。下一轮 decode 要处理 `14`，所以必须先给它准备新的 KV cache block。

### prefix cache

`prefix cache` 是复用相同 prompt 前缀的已计算 KV cache。

```text
请求 A: [10, 11, 12, 13, 99]
请求 B: [10, 11, 12, 13, 88]
```

如果 `block_size = 4`，两条请求的第一个完整 block 一样：

```text
[10, 11, 12, 13]
```

请求 B 可以复用请求 A 已经算好的这个 block 的 KV cache。这样 B 不必重新 prefill 这 4 个 token。

源码里的配合：

```text
hash_blocks(seq): 把已经计算过的完整 prefix block 登记到 hash_to_block_id。
can_allocate(seq): 新请求进来时从前往后检查完整 block 是否命中 prefix cache。
allocate(seq, num_cached_blocks): 对命中的 block 增加 ref_count，对未命中的部分新分配 block。
```

链式 hash 的意义：

```text
block1 的 hash 不只包含 block1 自己，还包含前一个 block 的 hash。
这样可以保证命中的是“从开头到这里都一样”的前缀，而不是某个局部 block 偶然一样。
```

### ref_count 与 deallocate

有了 prefix cache，一个物理 block 可能被多条 Sequence 同时引用：

```text
A.block_table = [3, 7]
B.block_table = [3, 8]
```

其中 block 3 是共享前缀。它的 `ref_count = 2`。

当 A 结束时：

```text
block 3.ref_count: 2 -> 1
block 7.ref_count: 1 -> 0
```

block 3 不能释放，因为 B 还在用。只有 `ref_count == 0` 的 block 才能回到 `free_block_ids`。

一句话：

```text
ref_count 是为了安全复用 prefix cache block；deallocate 只释放没人再引用的 block。
```

### can_append、may_append 与 preempt

decode 前，scheduler 会先问：

```python
self.block_manager.can_append(seq)
```

如果这轮 decode 不需要新 block，永远可以 append。如果需要新 block，就必须确认 `free_block_ids` 至少还有 1 个。

如果 block 不够，scheduler 会抢占某些 running 请求：

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

抢占不是丢弃请求，而是释放它的 KV cache，把它退回 waiting。因为 KV cache 已经被释放，它之后不能直接 decode，必须重新 prefill 来重建上下文 KV cache。

这里的 `seq.is_prefill = True` 表示：

```text
这条 Sequence 下一次被模型处理时，要按 prefill 模式处理。
```

它不是说这条请求从未生成过 token，而是说它的 GPU KV cache 需要重新建立。

### block_table 进入模型执行

`block_table` 不只是 Python 侧资源管理用的。`ModelRunner` 会把多条请求的 `block_table` 组织成 `block_tables`，并生成 `slot_mapping`。

```text
block_tables:
给 attention 查历史 KV cache 用，说明每条 sequence 的逻辑 block 在哪些物理 block。

slot_mapping:
给 store_kvcache 写当前 token 的 K/V 用，说明本轮每个 token 的 KV cache 写到哪个具体槽位。
```

所以完整链路是：

```text
Scheduler 选择哪些 seq 跑
BlockManager 确保 KV cache block 足够
ModelRunner 根据 block_table 准备 block_tables 和 slot_mapping
Attention 读写 KV cache
Sampler 采样下一个 token
Scheduler.postprocess 更新 Sequence 并在结束时释放 block
```

### 本节压缩总结

```text
BlockManager 负责把 Sequence 的 token 序列映射到 GPU KV cache block；
它能为 prefill 一次分配多个 block，为 decode 按需追加 block；
还能把已计算的完整前缀 block 做 hash 登记，让后来的相同前缀请求复用；
最后通过 ref_count 保证共享 block 不会被提前释放。
```
