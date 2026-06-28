# 学习进度

Last updated: 2026-06-28

## Learner Profile

- 已具备：基础 Python、基础 C++。
- 需要补齐：机器学习/深度学习最小基础、Transformer、LLM 推理流程、GPU/CUDA 基础、PyTorch 张量执行、KV cache、continuous batching、PagedAttention 思想、vLLM 调度、采样、张量并行、CUDA Graph、性能分析。
- 学习偏好：从 nano-vllm 项目出发，概念解释必须落到源码和可运行练习。
- 补充要求：除了本项目内容，需要穿插讲解必要前置知识。采用“双轨学习”：源码主线负责建立真实系统理解，前置知识支线负责补齐读源码所需的概念。
- 总目标：从零基础入手，通过 nano-vllm 掌握 vLLM，进而掌握大模型推理的基本技能。
- 结束学习约定：当学习者说“不学了”“今天到这”“停止学习”等表达时，先更新本进度文件，再提交学习相关改动并推送到 `origin/main`。
- 教学详细度要求：讲解需要更细，默认多补充背景知识、动机、定义、例子和源码锚点。
- 讲义维护要求：结合教学内容和学习者提问，随时更新 `learning/docs/` 中每一课的讲义，使其持续沉淀为个人复习材料。
- 教学顺序要求：必须先讲背景知识和核心直觉，再引入 nano-vllm 源码，最后用源码回扣知识点；不要一上来就看源码。
- 代码操练要求：每一课尽量安排实际代码编写练习，优先放在 `learning/exercises/`，并在讲义中记录目标、运行方式、预期观察和思考题。

## Current Position

- Stage: 0 - 学习档案与项目地图搭建
- Current lesson: `learning/docs/lesson_01_scheduler.md`
- Status: scheduler_schedule_understood
- Last covered: 学习者已经理解 nano-vllm 的 `Scheduler.schedule()`：waiting/running 队列、prefill 优先、chunked prefill 限制、`num_tokens`/`num_scheduled_tokens`、prefix cache 命中、`Sequence.num_tokens` 与 `num_prompt_tokens` 的区别。

## Lesson Status

- `learning/docs/prereq_minus1_from_zero_to_inference.md`: recap_passed
- `learning/docs/prereq_00_llm_inference_big_picture.md`: mostly_passed_needs_short_review
- `learning/docs/lesson_00_project_map.md`: pending_restart
- `learning/docs/lesson_01_scheduler.md`: schedule_function_understood

说明：学习者要求“从最开始，重新学习”。此前内容不删除，作为历史材料和后续讲义保留；但当前学习路径从零基础入口重新开始。后续必须按“背景问题 -> 核心直觉 -> 小例子 -> 源码 -> 回扣知识”的顺序推进。

## Next Lesson

继续 `learning/docs/lesson_01_scheduler.md`。目标是从已经理解的 `schedule()` 继续进入 `postprocess()`、`BlockManager` 和 KV cache block 生命周期。

## Next Actions

1. 从 `learning/docs/lesson_01_scheduler.md` 继续。
2. 先让学习者用自己的话复述：`schedule()` 如何先 prefill、再 decode，以及为什么本轮有 prefill 就直接返回。
3. 进入 `Scheduler.postprocess()`：解释模型输出 token 后如何更新 `num_cached_tokens`、追加 token、判断 finished、释放 KV cache。
4. 温和引入 `nanovllm/engine/block_manager.py`：只讲 block 分配、释放、prefix cache 的最小必要背景。
5. 准备一个小练习：扩展 toy scheduler，显式打印“请求完成后释放资源”的动作，再对照真实 `block_manager.deallocate(seq)`。

## Session Log

### 2026-06-26

- 新增 `learning/` 学习目录。
- 新增 `AGENTS.md`，约定后续在本仓库中听到“开始学习/继续学习”时读取本进度文件并接着教。
- 新增长期路线图、第一课讲义和第一课练习。
- 根据学习者补充要求，新增“前置知识支线”，后续会在源码课之前穿插讲解必要背景。
- 根据学习者目标，明确默认从零基础开始，最终目标是通过 nano-vllm 掌握 vLLM 和大模型推理基本技能。
- 根据学习者要求，新增结束学习时自动更新进度、提交并推送到 `origin/main` 的约定。
- 开始第一次学习：讲解从零到推理技能地图、LLM 推理大图，运行 `python3 learning/exercises/lesson_00_sequence_blocks.py`，观察 prompt token、completion token 和 block table 的增长。
- 根据学习者补充要求，明确后续讲解需要更详细、更多背景，并且每次根据教学内容和提问持续更新 `learning/docs/` 讲义。
- 继续 Lesson 00：详细讲解 `LLMEngine.__init__()`、`add_request()`、`step()`、`generate()` 的职责，把“外部 prompt 到内部 Sequence，再到 step 心跳”的主路径写入 `learning/docs/lesson_00_project_map.md`。
- 根据学习者反馈，修正教学方法：后续必须先讲背景知识，再从知识引入源码，最后用源码回扣知识，而不是直接源码优先。
- 开始 Lesson 01：新增 `learning/docs/lesson_01_scheduler.md`，讲解 scheduler 背景、waiting/running 队列、prefill 优先、decode 合批；新增并验证 `learning/exercises/lesson_01_toy_scheduler.py`。
- 澄清课程状态：Lesson 00 主体内容已完成，但还缺学习者复述确认；Lesson 01 不是跳课，而是从 Lesson 00 的 `scheduler.schedule()` 调用自然展开。
- 学习者要求从最开始重新学习：当前进度重置到 `learning/docs/prereq_minus1_from_zero_to_inference.md`，此前 Lesson 00/01 材料保留为历史讲义和后续复用内容。
- 学习者补充要求：每一课尽量安排实际代码编写操练；后续讲义和学习流程需要包含代码练习目标、运行方式、预期观察和思考题。
- 重新从头学习：讲解训练和推理的区别、推理系统关注点、自回归生成循环；运行 `python3 learning/exercises/prereq_00_toy_autoregressive.py`；补充 `learning/docs/prereq_minus1_from_zero_to_inference.md` 的参数/训练/推理说明和练习观察。
- 学习者通过第一组检查点：训练会更新模型、推理不会；推理逐 token 循环生成；context 每步累积历史结果。进入 `prereq_00_llm_inference_big_picture.md`，新增 `learning/exercises/prereq_01_toy_tokenizer_sampler.py`。
- 结束本次学习：讲解 token/tokenizer、logits/sampler；验证 `python3 learning/exercises/prereq_01_toy_tokenizer_sampler.py`；下次从修改 `TOY_LOGITS` 的代码操练和 prompt/completion/context 继续。

## Open Questions

- 你本机是否有可用 NVIDIA GPU，以及是否已经下载 Qwen3-0.6B 权重。
- 你希望学习节奏偏“每天短课”还是“每次深入 1-2 小时”。
## Session 2026-06-27

### Summary

本次按学习者反馈调整教学方式：先讲背景和直觉，再进入练习输出，暂不急着深入 nano-vllm 源码。学习者明确指出上一轮讲解节奏过快、把代理运行的练习误说成“你跑通了”，后续教学必须保持教程风格、循序渐进。

### Concepts Covered

- 自回归生成：LLM 不是一次性生成完整答案，而是不断预测下一个 token，并把它追加回上下文。
- tokenizer：文本给人看，token id 给模型做数字计算。
- logits：模型给候选 next token 的分数，不是最终文本。
- greedy sampler：选择 logits 分数最高的 token id。
- decode 输出复盘：学习者亲自运行 `prereq_01_toy_tokenizer_sampler.py`，能解释为什么选择 `next_id=5` 而不是 `next_id=7`。
- prefill 与 decode：
  - prefill 处理用户一开始给出的整段 prompt。
  - decode 每轮处理一个新生成 token。
- KV cache：
  - 用显存空间换时间，缓存旧 token 的中间结果，避免 decode 每轮从头重复计算。
  - 学习者能回答没有 KV cache 会造成计算资源浪费，长 prompt 和多用户会带来显存压力。
- scheduler：
  - 在有限 GPU/KV cache/token batch 资源下，决定每轮哪些请求 prefill、哪些请求 decode、哪些请求等待或释放资源。
  - 学习者能判断结束请求需要释放 cache，prefill 未完成的请求仍属于 waiting。

### Files / Exercises Used

- `learning/exercises/prereq_01_toy_tokenizer_sampler.py`
- `learning/exercises/lesson_01_toy_scheduler.py`
- `learning/docs/prereq_00_llm_inference_big_picture.md`
- `learning/docs/lesson_01_scheduler.md`

### Learner Answers / Checks

- 能回答 greedy sampler 选择“分数最高”的 token。
- 能说明 100 个 prompt token 属于 prefill，后续 20 个生成 token 属于 decode。
- 能指出没有 KV cache 会浪费计算资源。
- 能指出 KV cache 在长 prompt、多请求情况下带来显存压力。
- 能解释 toy scheduler 中 B 已经进入 prefill batch 但仍在 waiting，是因为 B 还没有 prefill 完成。

### Lesson Docs Updated

- `learning/docs/prereq_00_llm_inference_big_picture.md`：补充 2026-06-27 课堂复盘，强调教学节奏和概念边界。
- `learning/docs/lesson_01_scheduler.md`：补充 toy scheduler 输出解释，强调 waiting/running 的真实含义。

### Current Lesson Status

- `learning/docs/prereq_minus1_from_zero_to_inference.md`: recap_passed
- `learning/docs/prereq_00_llm_inference_big_picture.md`: mostly_passed_needs_short_review
- `learning/docs/lesson_01_scheduler.md`: in_progress
- `learning/docs/lesson_00_project_map.md`: pending_restart

### Next Action

下次从 `learning/docs/lesson_01_scheduler.md` 继续。先用 5 分钟复述三层主线：

```text
单请求生成 -> prefill/decode 与 KV cache -> 多请求 scheduler
```

然后进入 nano-vllm 源码的第一批温和锚点：

- `nanovllm/engine/sequence.py`：请求状态、prompt token、generated token、finished。
- `nanovllm/engine/scheduler.py`：waiting/running 队列、资源限制、调度决策。
- `nanovllm/engine/llm_engine.py`：`step()` 如何把 scheduler 和 model runner 串起来。

## Session 2026-06-28

### Summary

本次开始前按新约束先从 `origin/main` 快进更新代码，然后继续 `lesson_01_scheduler.md`。学习重点从 toy scheduler 过渡到 nano-vllm 的真实 `Scheduler.schedule()`，并围绕学习者的问题逐段澄清调度细节。

### Concepts Covered

- `schedule()` 的返回值：`list[Sequence]` 和 `is_prefill`。
- prefill 优先：本轮只要安排到 prefill batch，就返回，不混合 decode。
- waiting/running 状态变化：prefill 完成后从 `waiting` 移到 `running`。
- chunked prefill 限制：允许一个 prefill batch 有多个请求，但只允许 batch 的第一条请求被切开。
- `num_tokens`：当前 seq 在 prefill 阶段还需要模型处理的 token 数。
- `num_scheduled_tokens`：本轮实际安排处理的 token 数。
- prefix cache 命中：相同 prompt 前缀的完整 KV cache block 可以复用，减少需要重新 prefill 的 token。
- `Sequence.num_tokens` 与 `num_prompt_tokens`：初始化时相同，但前者随生成增长，后者固定记录原始 prompt 长度。
- toy scheduler 与真实源码的区别：toy 中请求完成后“不再 append 回 running”表示释放资源；真实 nano-vllm 中 `postprocess()` 会调用 `block_manager.deallocate(seq)`。

### Files / Exercises Used

- `learning/exercises/lesson_01_toy_scheduler.py`
- `learning/docs/lesson_01_scheduler.md`
- `nanovllm/engine/sequence.py`
- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/llm_engine.py`
- `nanovllm/engine/block_manager.py`

### Learner Answers / Checks

- 能解释 `waiting` 存放未完成 prefill 的请求，`running` 存放已经完成 prefill、还未 decode 结束的请求。
- 能解释 `max_num_batched_tokens` 变小会让长 prompt 分更多轮 prefill。
- 能指出 step 3 中 `[B, C]` 是因为 B 剩余 2 个 token、C 需要 2 个 token，都能完整放入剩余 budget。
- 能理解 `schedule()` 的整体逻辑，并明确表示已经理解 nano-vllm 的 `schedule()` 函数。

### Lesson Docs Updated

- `learning/docs/lesson_01_scheduler.md`：补充 prefix cache、chunked prefill 限制、toy 释放资源与真实 `deallocate()` 的区别、`Sequence` 字段语义、`Scheduler.schedule()` 逐段解释。

### Current Lesson Status

- `learning/docs/prereq_minus1_from_zero_to_inference.md`: recap_passed
- `learning/docs/prereq_00_llm_inference_big_picture.md`: mostly_passed_needs_short_review
- `learning/docs/lesson_01_scheduler.md`: schedule_function_understood
- `learning/docs/lesson_00_project_map.md`: pending_restart

### Next Action

下次继续 `learning/docs/lesson_01_scheduler.md`。先快速复述 `schedule()`，然后进入 `Scheduler.postprocess()` 和 `BlockManager` 的最小必要背景：模型输出 token 后如何更新 cache 进度、追加 token、判断 finished，并释放 KV cache block。
