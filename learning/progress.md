# 学习进度

Last updated: 2026-06-26

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
- Current lesson: `learning/docs/prereq_00_llm_inference_big_picture.md`
- Status: restarted_from_beginning
- Last covered: 学习者正确复述训练/推理区别、自回归循环和 context 累积；已进入 LLM 推理大图，讲解 token/tokenizer、logits/sampler，并新增、验证极简 tokenizer/sampler 代码操练。

## Lesson Status

- `learning/docs/prereq_minus1_from_zero_to_inference.md`: recap_passed
- `learning/docs/prereq_00_llm_inference_big_picture.md`: in_progress
- `learning/docs/lesson_00_project_map.md`: pending_restart
- `learning/docs/lesson_01_scheduler.md`: created_for_later

说明：学习者要求“从最开始，重新学习”。此前内容不删除，作为历史材料和后续讲义保留；但当前学习路径从零基础入口重新开始。后续必须按“背景问题 -> 核心直觉 -> 小例子 -> 源码 -> 回扣知识”的顺序推进。

## Next Lesson

继续 `learning/docs/prereq_00_llm_inference_big_picture.md`。目标是理解 token/tokenizer、prompt/completion、logits/sampler，并通过代码操练跑通文本到 token id 再到采样生成的最小链路。

## Next Actions

1. 从 `learning/docs/prereq_00_llm_inference_big_picture.md` 继续。
2. 先回顾 token/tokenizer、logits/sampler：模型读 token id，不直接读字符串；模型输出 logits，sampler 负责选 next token。
3. 让学习者运行并修改 `python3 learning/exercises/prereq_01_toy_tokenizer_sampler.py`，例如调整 `TOY_LOGITS` 第一步分数，观察生成结果变化。
4. 继续讲 prompt/completion/context，然后引入 prefill/decode 的背景。
5. 回扣到 nano-vllm：后续会在 `LLMEngine.add_request()`、`SamplingParams`、`Sampler` 中看到同样概念。

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
