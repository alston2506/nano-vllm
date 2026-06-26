"""Prerequisite exercise: a tiny autoregressive generation loop.

Run from the repository root:

    python3 learning/exercises/prereq_00_toy_autoregressive.py
"""


NEXT_TOKEN = {
    "我": "想",
    "想": "学习",
    "学习": "大模型",
    "大模型": "推理",
    "推理": "系统",
    "系统": "。",
}


def choose_next_token(context: list[str]) -> str:
    """Pretend to be a language model that predicts one next token."""
    last_token = context[-1]
    return NEXT_TOKEN.get(last_token, "。")


def generate(prompt_tokens: list[str], max_new_tokens: int) -> list[str]:
    context = list(prompt_tokens)
    for step in range(max_new_tokens):
        next_token = choose_next_token(context)
        context.append(next_token)
        print(f"step {step + 1}: next={next_token}, context={''.join(context)}")
        if next_token == "。":
            break
    return context


def main() -> None:
    prompt_tokens = ["我"]
    print(f"prompt: {''.join(prompt_tokens)}")
    output_tokens = generate(prompt_tokens, max_new_tokens=8)
    print(f"final:  {''.join(output_tokens)}")


if __name__ == "__main__":
    main()
