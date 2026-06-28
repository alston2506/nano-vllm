"""Lesson 01 exercise: simulate a tiny inference scheduler.

Run from the repository root:

    python3 learning/exercises/lesson_01_toy_scheduler.py
"""

from collections import deque
from dataclasses import dataclass


@dataclass
class ToyRequest:
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    cached_prompt_tokens: int = 0
    generated_tokens: int = 0

    @property
    def prefill_done(self) -> bool:
        return self.cached_prompt_tokens == self.prompt_tokens

    @property
    def finished(self) -> bool:
        return self.generated_tokens == self.max_new_tokens


class ToyScheduler:
    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[ToyRequest] = deque()
        self.running: deque[ToyRequest] = deque()

    def add(self, request: ToyRequest) -> None:
        self.waiting.append(request)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def step(self) -> None:
        scheduled: list[ToyRequest] = []
        used_tokens = 0

        while self.waiting and len(scheduled) < self.max_num_seqs:
            request = self.waiting[0]
            remaining_budget = self.max_num_batched_tokens - used_tokens
            remaining_prompt = request.prompt_tokens - request.cached_prompt_tokens
            if remaining_budget == 0:
                break
            if remaining_prompt > remaining_budget and scheduled:
                break

            to_cache = min(remaining_prompt, remaining_budget)
            request.cached_prompt_tokens += to_cache
            used_tokens += to_cache
            scheduled.append(request)

            if request.prefill_done:
                self.waiting.popleft()
                self.running.append(request)

        if scheduled:
            ids = ", ".join(req.request_id for req in scheduled)
            print(f"prefill batch: [{ids}], prompt tokens processed={used_tokens}")
            self.print_queues()
            return

        while self.running and len(scheduled) < self.max_num_seqs:
            request = self.running.popleft()
            request.generated_tokens += 1
            scheduled.append(request)
            if not request.finished:
                self.running.append(request)

        ids = ", ".join(req.request_id for req in scheduled)
        print(f"decode batch:  [{ids}], one token per request")
        self.print_queues()

    def print_queues(self) -> None:
        waiting = ", ".join(req.request_id for req in self.waiting) or "-"
        running = ", ".join(req.request_id for req in self.running) or "-"
        print(f"  waiting: [{waiting}]")
        print(f"  running: [{running}]")


def main() -> None:
    scheduler = ToyScheduler(max_num_seqs=2, max_num_batched_tokens=3)
    scheduler.add(ToyRequest("A", prompt_tokens=4, max_new_tokens=3))
    scheduler.add(ToyRequest("B", prompt_tokens=8, max_new_tokens=2))
    scheduler.add(ToyRequest("C", prompt_tokens=2, max_new_tokens=4))

    step = 1
    while not scheduler.is_finished():
        print(f"\n== step {step} ==")
        scheduler.step()
        step += 1


if __name__ == "__main__":
    main()
