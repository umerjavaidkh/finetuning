from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv

from .schema_adapter import SFTPromptCandidate, TaskType

load_dotenv()

SYSTEM_PROMPT = (
    "أنت مساعد تربوي متخصص في المناهج الدراسية العربية. تُنتج مواد تعليمية "
    "بالفصحى ملتزمة ببنية المنهاج ومؤشرات التعلّم. لا تخترع حقائق منهجية "
    "غير واردة في السياق المرفق."
)

_TASK_INSTRUCTIONS: dict[TaskType, str] = {
    TaskType.EXAM: (
        "بالاعتماد على السياق أدناه فقط، أنشئ امتحانًا قصيرًا (٤-٦ أسئلة متنوعة) "
        "مع مفتاح إجابة كامل في نهاية الامتحان."
    ),
    TaskType.WORKSHEET: (
        "بالاعتماد على السياق أدناه فقط، أنشئ ورقة عمل للتلاميذ تتضمن تعليمات "
        "واضحة وأنشطة عملية تناسب المستوى الدراسي المذكور."
    ),
    TaskType.INDICATOR_QUESTIONS: (
        "بالاعتماد على مؤشرات التعلّم المذكورة في السياق أدناه، أنشئ مجموعة أسئلة "
        "مرتبطة مباشرة بكل مؤشر، مع الإجابة النموذجية لكل سؤال."
    ),
    TaskType.LESSON_PLAN: (
        "بالاعتماد على السياق أدناه، أنشئ خطة شرح للدرس تتضمن الهدف، والخطوات "
        "الإجرائية للمعلّم، والوقت المقدّر لكل خطوة."
    ),
    TaskType.GRAMMAR_EXPLANATION: (
        "بالاعتماد على السياق أدناه، اشرح القاعدة النحوية أو الصرفية المشار إليها "
        "بأسلوب مناسب لمستوى الصف المذكور، مع أمثلة توضيحية."
    ),
    TaskType.VOCAB_ACTIVITY: (
        "بالاعتماد على السياق أدناه، أنشئ نشاط مفردات (جذور، اشتقاقات، استخدام في "
        "سياق جديد) للكلمات الواردة في النص."
    ),
    TaskType.STORY_COMPREHENSION: (
        "بالاعتماد على عناصر القصة المذكورة في السياق أدناه، أنشئ أسئلة استيعاب "
        "قصصي تركّز على الشخصيات والمشكلة والحل، مع إجابات نموذجية."
    ),
}


class TeacherModelClient(Protocol):
    def generate(self, system: str, user: str) -> str: ...


@dataclass
class OpenAIChatClient:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    temperature: float = 0.7

    def __post_init__(self) -> None:
        resolved_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set and no api_key provided to OpenAIChatClient"
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=resolved_key)

    def generate(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content


def build_user_prompt(candidate: SFTPromptCandidate) -> str:
    instruction = _TASK_INSTRUCTIONS[candidate.task_type]
    return f"{instruction}\n\n{candidate.context_block}"


def generate_pair(candidate: SFTPromptCandidate, client: TeacherModelClient) -> dict:
    user_prompt = build_user_prompt(candidate)
    assistant_output = client.generate(SYSTEM_PROMPT, user_prompt)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_output},
        ],
        "task_type": candidate.task_type.value,
        "source_entry_id": candidate.source_entry_id,
        "doc_id": candidate.doc_id,
        "unit": candidate.unit,
        "lesson": candidate.lesson,
        "lesson_title": candidate.lesson_title,
        "oversample_weight": candidate.oversample_weight,
    }


def generate_dataset(
    candidates: list[SFTPromptCandidate], client: TeacherModelClient
) -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    failures: list[dict] = []
    for candidate in candidates:
        try:
            results.append(generate_pair(candidate, client))
        except Exception as e:
            failures.append(
                {
                    "source_entry_id": candidate.source_entry_id,
                    "task_type": candidate.task_type.value,
                    "error": str(e),
                }
            )
    return results, failures
