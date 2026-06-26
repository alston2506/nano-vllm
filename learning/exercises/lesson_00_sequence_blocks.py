"""Lesson 00 exercise: simulate Sequence blocks without loading a model.

Run from the repository root:

    python3 learning/exercises/lesson_00_sequence_blocks.py
"""

from dataclasses import dataclass, field


@dataclass
class ToySequence:
    prompt_token_ids: list[int]
    block_size: int = 4
    token_ids: list[int] = field(init=False)
    num_prompt_tokens: int = field(init=False)
    num_cached_tokens: int = 0
    block_table: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.token_ids = list(self.prompt_token_ids)
        self.num_prompt_tokens = len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    def block(self, index: int) -> list[int]:
        start = index * self.block_size
        end = start + self.block_size
        return self.token_ids[start:end]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)


class ToyBlockManager:
    def __init__(self) -> None:
        self.next_block_id = 0

    def allocate_missing_blocks(self, seq: ToySequence) -> None:
        while len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self.next_block_id)
            self.next_block_id += 1


def print_sequence_state(title: str, seq: ToySequence) -> None:
    print(f"\n== {title} ==")
    print(f"tokens: {seq.token_ids}")
    print(f"prompt tokens: {seq.prompt_token_ids}")
    print(f"completion tokens: {seq.completion_token_ids}")
    print(f"num_tokens: {seq.num_tokens}")
    print(f"num_blocks: {seq.num_blocks}")
    print(f"block_table: {seq.block_table}")
    for i, block_id in enumerate(seq.block_table):
        print(f"  logical block {i} -> physical block {block_id}, tokens={seq.block(i)}")


def main() -> None:
    prompt = [101, 42, 17, 9, 88, 12]
    generated_tokens = [2001, 2002, 2003, 2004, 2005]

    seq = ToySequence(prompt_token_ids=prompt, block_size=4)
    block_manager = ToyBlockManager()

    print_sequence_state("new request before prefill", seq)

    block_manager.allocate_missing_blocks(seq)
    seq.num_cached_tokens = seq.num_tokens
    print_sequence_state("after prefill cached the prompt", seq)

    for token_id in generated_tokens:
        seq.append_token(token_id)
        block_manager.allocate_missing_blocks(seq)
        seq.num_cached_tokens = seq.num_tokens
        print_sequence_state(f"after decode appended token {token_id}", seq)


if __name__ == "__main__":
    main()
