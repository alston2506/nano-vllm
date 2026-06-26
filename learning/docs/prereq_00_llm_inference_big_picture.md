# Prerequisite 00: LLM 推理大图

这份前置讲义的目标是让你在读 nano-vllm 前，先知道一个大模型推理系统到底在做什么。

## 1. LLM 推理是什么

训练时，模型学会根据前面的 token 预测下一个 token。推理时，我们把 prompt 输入模型，模型不断预测下一个 token，再把这个新 token 接回输入后面，继续预测下一个。

这叫自回归生成。

简化流程：

```text
prompt -> tokenizer -> token ids -> model -> logits -> sampler -> next token
                                                     ^                 |
                                                     |                 v
                                             previous tokens <- append token
```

## 2. Tokenizer

模型不直接读字符串，而是读整数 token id。

例如一句话可能被 tokenizer 转成：

```text
"Hello, Nano-vLLM" -> [9707, 11, 12345, 678]
```

在 nano-vllm 中：

- `LLMEngine.__init__()` 创建 tokenizer。
- `LLMEngine.add_request()` 调用 `self.tokenizer.encode(prompt)`。
- `LLMEngine.generate()` 最后调用 `self.tokenizer.decode(token_ids)`。

### 更细一点：为什么一定要 token id

神经网络擅长处理数字张量，不擅长直接处理 Python 字符串。tokenizer 的作用就是建立一套约定：

```text
文本片段 <-> 整数 id
```

例如：

```text
我 -> 1
想 -> 2
学习 -> 3
```

真实 tokenizer 会复杂得多，可能把一个词拆成多个子词，也可能把中文、英文、标点混合编码。但第一层直觉很简单：模型看到的是 id，不是原始字符串。

## 3. Prompt 和 Completion

- prompt：用户输入的部分。
- completion：模型生成的部分。
- context：prompt + 已经生成的 completion。

在 `Sequence` 中：

- `num_prompt_tokens` 记录 prompt 长度。
- `completion_token_ids` 返回生成部分。
- `token_ids` 保存 prompt 和 completion 的整体上下文。

## 4. Logits 和采样

模型每一步输出的是 logits，可以理解为“每个词成为下一个 token 的分数”。采样器把 logits 转成一个具体 token。

常见采样参数：

- temperature：温度越高，输出越随机；越低，越确定。
- top-k：只在分数最高的 k 个 token 中选。
- top-p：只在累计概率达到 p 的候选 token 中选。

nano-vllm 目前先从 temperature 入手，对应：

- `nanovllm/sampling_params.py`
- `nanovllm/layers/sampler.py`

### 更细一点：logits 不是最终文本

模型 forward 的输出不是“下一个字”，而是一组分数：

```text
想: 9.0
学习: 1.0
推理: 0.5
```

采样器负责把这组分数变成一个 token。最简单的策略是 greedy sampling：永远选分数最高的 token。更灵活的策略会引入 temperature、top-k、top-p，让生成有随机性。

## 代码操练：极简 tokenizer 和 sampler

运行：

```bash
python3 learning/exercises/prereq_01_toy_tokenizer_sampler.py
```

这个练习模拟：

```text
文本 -> token ids -> toy logits -> greedy sampler -> next token id -> decode
```

练习目标：

- 理解 tokenizer 为什么把文本变成 id。
- 理解 logits 是候选 token 的分数，不是最终文本。
- 理解 sampler 如何从 logits 中选择一个 next token。
- 为后面阅读 `AutoTokenizer`、`SamplingParams`、`Sampler` 做准备。

可以尝试修改：

- `VOCAB`：加入新的 token。
- `TOY_LOGITS`：改变某一步的分数，观察生成结果怎么变。
- `greedy_sample()`：尝试不选最高分，而是选最低分，看看结果如何变奇怪。

## 5. 为什么有 Prefill 和 Decode

推理分成两个阶段：

### Prefill

处理 prompt。prompt 可能很长，所以这一阶段一次喂给模型很多 token。

结果：

- 得到第一个可采样的 next token。
- 同时把 prompt 对应的 K/V 写进 KV cache。

### Decode

逐个生成 completion token。每轮通常只输入上一步生成的一个 token。

结果：

- 通过 KV cache 复用历史上下文。
- 每轮生成一个新的 token。

## 6. 为什么 KV Cache 重要

Transformer attention 每生成一个 token，都需要看前面的上下文。如果每次都从头重新算 prompt 和历史 completion，会非常慢。

KV cache 的作用是缓存每层 attention 里已经算过的 key/value。decode 阶段只需要计算新 token 的 key/value，再和缓存里的历史 key/value 做 attention。

直觉：

- 没有 KV cache：每写一个字都重新读整本书。
- 有 KV cache：读过的内容做了索引，每次只处理新增的一点。

## 7. vLLM 解决的核心问题

LLM 推理服务不是只处理一个请求，而是同时处理很多请求。每个请求长度不同、生成速度不同、结束时间不同。

vLLM 关注的问题包括：

- 如何把多个请求动态组成 batch，提高 GPU 利用率？
- 如何管理大量 KV cache，避免显存浪费？
- 如何让请求来了就加入、结束了就释放，而不是等整个 batch 一起结束？
- 如何复用相同 prompt 前缀的缓存？

nano-vllm 是这些思想的轻量实现，适合学习。

## 8. 和 nano-vllm 的对应关系

- 请求状态：`nanovllm/engine/sequence.py`
- 推理主循环：`nanovllm/engine/llm_engine.py`
- 调度请求：`nanovllm/engine/scheduler.py`
- 管理 KV cache block：`nanovllm/engine/block_manager.py`
- 执行模型：`nanovllm/engine/model_runner.py`
- 采样：`nanovllm/layers/sampler.py`

## 检查点

你需要能用自己的话解释：

1. 为什么模型输入是 token id，不是字符串？
2. 自回归生成为什么要把新 token 接回上下文？
3. prefill 和 decode 的主要区别是什么？
4. KV cache 缓存的是什么，为什么能加速？
5. vLLM 类系统主要是在优化模型结构，还是在优化推理系统执行？
