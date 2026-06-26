# Prerequisite -1: 从零到大模型推理技能地图

这节课先不急着读源码。我们先建立一张地图：你要学的到底是什么，为什么 nano-vllm 能带你走到 vLLM，再走到大模型推理工程能力。

## 1. 你要掌握的不是一个库，而是一套系统能力

大模型推理的目标是：给定一个已经训练好的模型和用户输入，让模型又快、又稳、又省显存地生成输出。

这件事横跨几层知识：

- 模型层：Transformer 是怎么根据上下文预测下一个 token 的。
- 张量计算层：PyTorch 如何把矩阵运算交给 GPU。
- 系统层：多个请求如何排队、合批、抢占、释放资源。
- 内存层：KV cache 如何占用显存，如何被分块管理。
- 服务层：如何在吞吐、延迟、显存之间做取舍。

vLLM 是一个成熟推理引擎。nano-vllm 是它的轻量学习版：代码少，结构完整，适合从零拆开看。

## 2. 训练和推理有什么区别

训练是在更新模型参数：

```text
数据 -> 模型 -> 损失 -> 反向传播 -> 更新参数
```

推理不更新参数：

```text
prompt -> 模型 -> 下一个 token -> 追加到上下文 -> 继续生成
```

所以我们学习 nano-vllm 时，重点不是反向传播、优化器、数据集训练，而是：

- 模型结构如何执行 forward。
- token 如何逐个生成。
- KV cache 如何复用历史计算。
- scheduler 如何让多个请求共享 GPU。
- 显存如何被规划和回收。

### 更细一点：参数、训练、推理

可以先把模型想成一个很大的函数：

```text
输出 = 模型(输入, 参数)
```

参数是模型内部大量数字，比如权重矩阵。训练阶段会修改这些数字，让模型越来越会预测；推理阶段通常不修改这些数字，只使用它们。

所以：

- 训练像“学习做题方法”：做错了就改参数。
- 推理像“拿已经学会的方法做题”：不改参数，只尽快给答案。

学习 vLLM/nano-vllm 时，我们主要关心推理阶段。也就是说，我们关心的是如何把一个已经训练好的模型高效跑起来，而不是如何训练出这个模型。

### 推理系统关心什么

推理系统不是只调用一次模型。它还要负责：

- 把字符串变成 token id。
- 保存生成过程中的上下文。
- 每轮调用模型预测下一个 token。
- 把很多请求合成 batch，提高 GPU 利用率。
- 管理 KV cache，避免显存浪费。
- 请求完成后释放资源。

这些就是后面 `Sequence`、`Scheduler`、`BlockManager`、`ModelRunner` 要解决的问题。

## 3. 为什么要先懂一点 Transformer

你不需要一开始就会推导所有公式，但必须知道：

- LLM 的输入是 token id。
- token id 会变成向量。
- attention 让每个 token 读取前文信息。
- causal mask 保证模型不能偷看未来 token。
- 最后一层输出 logits，用来选下一个 token。

这些概念会直接对应到 nano-vllm 的：

- `nanovllm/models/qwen3.py`
- `nanovllm/layers/attention.py`
- `nanovllm/layers/embed_head.py`
- `nanovllm/layers/sampler.py`

## 4. 为什么推理系统重点在 KV Cache

大模型生成时，历史上下文会越来越长。如果每生成一个 token 都重新计算整个上下文，成本会越来越高。

KV cache 缓存 attention 中已经算过的 key/value，让 decode 阶段只处理新增 token。

vLLM 的核心贡献之一，就是把 KV cache 管理得更像操作系统管理内存页：按块分配、按块回收、减少浪费、支持动态请求。

nano-vllm 对应的入口是：

- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/model_runner.py`

## 5. 你最终要形成的基本技能

学完第一阶段后，你应该能做到：

- 看懂一次 `LLM.generate()` 从输入到输出的路径。
- 解释 prefill 和 decode 的区别。
- 解释 KV cache 为什么加速，以及为什么消耗大量显存。
- 解释 scheduler 为什么能提高 GPU 利用率。
- 看懂 block table 是如何把 sequence 映射到 KV cache block 的。
- 能写小练习模拟调度、缓存和采样。
- 能在 nano-vllm 中加简单调试信息或小功能。

学完更深入阶段后，你应该能进一步做到：

- 对比 nano-vllm 和 vLLM 的关键设计。
- 理解 PagedAttention 和 continuous batching。
- 分析吞吐、延迟、显存之间的取舍。
- 为一个推理服务选择基础参数，例如最大 batch token 数、最大并发请求数、最大上下文长度。

## 6. 我们的学习顺序

不要先背所有概念。顺序会是：

1. 先建立推理大图。
2. 跑一个不需要 GPU 的小练习，理解 token、sequence、block。
3. 读 nano-vllm 的请求主循环。
4. 补 Transformer 最小知识，再读模型结构。
5. 补 KV cache，再读 attention 和 block manager。
6. 补调度系统，再读 scheduler。
7. 补 GPU/PyTorch 执行，再读 model runner。
8. 最后对照真正的 vLLM 概念做扩展。

## 代码操练：最小自回归生成

本课的第一个代码操练不依赖模型、不依赖 GPU。它只模拟一件事：

```text
根据当前上下文预测下一个 token -> 把 token 接回上下文 -> 继续预测
```

运行：

```bash
python3 learning/exercises/prereq_00_toy_autoregressive.py
```

你会看到每一步都只生成一个新 token，但上下文会逐步变长。

练习目标：

- 先用纯 Python 抓住“自回归生成是循环”这个核心直觉。
- 理解为什么真实推理系统需要保存上下文状态。
- 为后面理解 `Sequence.token_ids`、`append_token()`、prefill/decode 做准备。

可以尝试修改：

- `NEXT_TOKEN`：改变“模型”的预测规则。
- `prompt_tokens`：改变起始 prompt。
- `max_new_tokens`：观察最大生成长度如何限制循环。

### 运行观察

当前练习的输出类似：

```text
prompt: 我
step 1: next=想, context=我想
step 2: next=学习, context=我想学习
...
final:  我想学习大模型推理系统。
```

这个输出要抓住两点：

1. 每一步只产生一个 `next`。
2. 每一步都会把 `next` 接回 `context`。

真实 LLM 的 `choose_next_token()` 不会像练习里这样查字典，而是通过 Transformer forward 得到 logits，再由 sampler 选 token。但外层循环的形状是一样的。

## 检查点

你需要能用自己的话回答：

1. 大模型训练和推理最核心的区别是什么？
2. 为什么学推理系统时，KV cache 和 scheduler 比反向传播更重要？
3. nano-vllm 和 vLLM 的关系是什么？
4. 你现在要掌握的是“模型算法”，还是“推理系统工程”，或者两者的哪一部分？
