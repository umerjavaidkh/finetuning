import pytest

from datagen.generate import (
    OpenAIChatClient,
    SYSTEM_PROMPT,
    build_user_prompt,
    generate_dataset,
    generate_pair,
)
from datagen.schema_adapter import SFTPromptCandidate, TaskType


class FakeClient:
    def __init__(self, response: str = "مخرج تجريبي من النموذج المعلّم.", fail_on=()):
        self.response = response
        self.fail_on = fail_on
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if len(self.calls) in self.fail_on:
            raise RuntimeError("simulated teacher model failure")
        return self.response


def make_candidate(task_type=TaskType.EXAM, source_entry_id="entry_1"):
    return SFTPromptCandidate(
        task_type=task_type,
        context_block="<السياق>\nنص تجريبي\n</السياق>",
        source_entry_id=source_entry_id,
        doc_id="doc-1",
        unit=1,
        lesson=3,
        lesson_title="درس تجريبي",
        content_type="objective",
        oversample_weight=2.0,
    )


def test_build_user_prompt_includes_task_instruction_and_context():
    candidate = make_candidate(task_type=TaskType.VOCAB_ACTIVITY)
    prompt = build_user_prompt(candidate)
    assert "نشاط مفردات" in prompt
    assert candidate.context_block in prompt


def test_every_task_type_has_an_instruction():
    for task_type in TaskType:
        candidate = make_candidate(task_type=task_type)
        prompt = build_user_prompt(candidate)
        assert len(prompt) > len(candidate.context_block)


def test_generate_pair_produces_chatml_shape():
    candidate = make_candidate()
    client = FakeClient(response="امتحان تجريبي مع مفتاح الإجابة.")
    pair = generate_pair(candidate, client)

    assert pair["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert pair["messages"][1]["role"] == "user"
    assert pair["messages"][2] == {
        "role": "assistant",
        "content": "امتحان تجريبي مع مفتاح الإجابة.",
    }
    assert pair["task_type"] == "exam"
    assert pair["source_entry_id"] == "entry_1"
    assert pair["doc_id"] == "doc-1"
    assert pair["oversample_weight"] == 2.0


def test_generate_dataset_collects_results_and_failures():
    candidates = [make_candidate(source_entry_id=f"entry_{i}") for i in range(4)]
    client = FakeClient(fail_on={2, 4})

    results, failures = generate_dataset(candidates, client)

    assert len(results) == 2
    assert len(failures) == 2
    assert failures[0]["source_entry_id"] == "entry_1"
    assert failures[0]["error"] == "simulated teacher model failure"


def test_openai_client_raises_fast_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIChatClient()
