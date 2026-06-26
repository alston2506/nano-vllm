# nano-vllm 长期学习路线图

这条路线的目标不是“看懂几段代码”，而是让你从零基础出发，能解释、修改、调试并扩展一个 vLLM 风格推理系统。

课程采用“双轨制”：

- 源码主线：围绕 nano-vllm 的真实实现推进。
- 前置知识支线：在读源码前补齐必要背景，学到能解释当前代码为止。

每节课都必须遵循背景优先：

`背景问题 -> 核心直觉 -> 小例子 -> nano-vllm 源码位置 -> 源码回扣知识 -> 代码操练 -> 复述检查 -> 更新讲义`

源码不是第一入口，而是用来验证和固定概念的材料。只有当背景问题和核心直觉已经讲清楚后，才进入对应文件和函数。

每课尽量包含一个实际代码编写操练。早期练习以纯 Python 小模拟为主，避免依赖 GPU 和模型权重；中后期逐渐过渡到调试脚本、单元测试、性能观察和小型源码改造。

## Lesson Docs Policy: 讲义随学随改

`learning/docs/` 中的每一课都是活文档。它们需要随着实际教学不断变厚，而不是只保留最初的大纲。

每次学习时，如果出现新的解释、类比、问题、误区、源码发现或练习结论，都要回写到对应讲义中。这样你以后复习时看到的不是通用教程，而是和你自己的学习路径贴合的材料。

每份讲义优先包含：

- 背景：为什么要学这个概念。
- 定义：最小但准确的解释。
- 例子：用小数据或小代码说明。
- 源码锚点：对应 nano-vllm 文件、类、函数。
- 代码操练：要写什么、怎么运行、观察什么。
- 常见误区：容易混淆的点。
- 检查点：你能否用自己的话说清楚。

## Prerequisite Track: 必要前置知识

这些知识不会一次性全部讲完，而是在源码需要时插入。

### P-1: 从零到大模型推理技能地图

- 机器学习、深度学习、Transformer、LLM、vLLM 分别是什么
- 训练和推理的区别
- 为什么学习推理系统不等于学习模型训练
- 你最终需要掌握的工程技能清单

对应讲义：`learning/docs/prereq_minus1_from_zero_to_inference.md`

### P0: LLM 推理大图

- token 和 tokenizer
- prompt、completion、上下文长度
- 自回归生成
- logits 和采样
- prefill 与 decode 的直觉

对应讲义：`learning/docs/prereq_00_llm_inference_big_picture.md`

### P1: Transformer 最小知识集

- embedding
- self-attention
- causal mask
- multi-head attention
- MLP
- layer norm 和 residual

### P2: PyTorch 与 GPU 执行直觉

- tensor shape
- dtype
- CPU/GPU 设备
- 显存占用
- kernel launch
- 为什么推理性能不是只看 Python 代码

### P3: LLM 推理系统基础

- batching
- latency vs throughput
- KV cache
- cache block
- 调度器
- 内存复用

### P4: vLLM 背景

- continuous batching
- PagedAttention
- prefix cache
- preemption
- tensor parallel

## Stage 0: 项目地图和推理总流程

- nano-vllm 目录结构
- `LLM.generate()` 的请求生命周期
- prompt token、completion token、prefill、decode
- `Sequence` 为什么是推理系统的最小调度单位

练习：用纯 Python 模拟 sequence 分块和生成过程。

## Stage 1: Transformer 与自回归生成基础

- tokenization
- embedding、attention、MLP、layer norm、residual
- causal LM 的 next-token prediction
- logits、temperature、采样

对应源码：

- `nanovllm/models/qwen3.py`
- `nanovllm/layers/`
- `nanovllm/layers/sampler.py`

## Stage 2: Prefill、Decode 与 KV Cache

- 为什么生成分成 prefill 和 decode
- attention 复杂度为什么会成为瓶颈
- KV cache 的形状、生命周期和显存占用
- block table 与 slot mapping

对应源码：

- `nanovllm/engine/model_runner.py`
- `nanovllm/layers/attention.py`
- `nanovllm/utils/context.py`

## Stage 3: vLLM 的核心系统思想

- dynamic batching / continuous batching
- PagedAttention 背景思想
- block manager
- prefix cache
- preemption

对应源码：

- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/sequence.py`

入门讲义：

- `learning/docs/lesson_01_scheduler.md`

## Stage 4: GPU 执行与性能

- PyTorch CUDA 张量执行
- FlashAttention
- CUDA Graph
- pinned memory 与 non-blocking copy
- 吞吐、延迟、显存三角关系

对应源码：

- `nanovllm/engine/model_runner.py`
- `nanovllm/layers/attention.py`
- `bench.py`

## Stage 5: 模型加载和权重并行

- Hugging Face config 和权重命名
- safetensors / PyTorch 权重加载
- tensor parallel 基础
- distributed process group

对应源码：

- `nanovllm/utils/loader.py`
- `nanovllm/layers/linear.py`
- `nanovllm/engine/model_runner.py`

## Stage 6: 实战改造

候选项目：

- 给 scheduler 增加更详细的调试日志。
- 写一个小型 profiler，输出 prefill/decode token 数和耗时。
- 给 `SamplingParams` 增加 top-k 或 top-p。
- 写单元测试验证 block manager 的 prefix cache 行为。
- 对比 nano-vllm 和 vLLM 的接口/概念差异。

## 学完后的能力目标

你应该能够：

- 画出一次离线推理请求在系统内的完整路径。
- 解释 prefill/decode、KV cache、block table、slot mapping、prefix cache。
- 独立定位 nano-vllm 中调度、缓存、采样和模型执行的代码。
- 设计并实现一个小型推理系统功能改造。
- 读懂 vLLM 文档和关键论文/博客中的核心系统概念。
