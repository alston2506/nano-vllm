"""Prerequisite exercise: tiny tokenizer and sampler.

Run from the repository root:

    python3 learning/exercises/prereq_01_toy_tokenizer_sampler.py
"""


VOCAB = {
    "<unk>": 0,
    "我": 1,
    "想": 2,
    "学习": 3,
    "大模型": 4,
    "推理": 5,
    "系统": 6,
    "。": 7,
}

ID_TO_TOKEN = {token_id: token for token, token_id in VOCAB.items()}

TOY_LOGITS = {
    1: {2: 9.0, 3: 1.0},
    2: {3: 8.0, 5: 2.0},
    3: {4: 7.0, 6: 1.0},
    4: {5: 8.0, 7: 1.0},
    5: {6: 8.0, 7: 2.0},
    6: {7: 10.0},
}


def encode(text: str) -> list[int]:
    """A toy tokenizer that splits by spaces."""
    return [VOCAB.get(piece, VOCAB["<unk>"]) for piece in text.split()]


def decode(token_ids: list[int]) -> str:
    return "".join(ID_TO_TOKEN.get(token_id, "<unk>") for token_id in token_ids)


def model_forward(context_ids: list[int]) -> dict[int, float]:
    """Pretend to be a model that returns logits for the next token."""
    last_token_id = context_ids[-1]
    return TOY_LOGITS.get(last_token_id, {VOCAB["。"]: 10.0})


def greedy_sample(logits: dict[int, float]) -> int:
    """Pick the token id with the highest score."""
    return max(logits, key=logits.get)


def generate(prompt: str, max_new_tokens: int) -> list[int]:
    context_ids = encode(prompt)
    print(f"prompt text: {prompt}")
    print(f"prompt ids:  {context_ids}")

    for step in range(max_new_tokens):
        logits = model_forward(context_ids)
        next_token_id = greedy_sample(logits)
        context_ids.append(next_token_id)
        print(
            f"step {step + 1}: logits={logits}, "
            f"next_id={next_token_id}, text={decode(context_ids)}"
        )
        if next_token_id == VOCAB["。"]:
            break

    return context_ids


def main() -> None:
    output_ids = generate("我", max_new_tokens=8)
    print(f"final ids:  {output_ids}")
    print(f"final text: {decode(output_ids)}")


if __name__ == "__main__":
    main()
