from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .generate import TeacherModelClient

RUBRIC_CRITERIA = (
    "language_correctness",
    "curriculum_fidelity",
    "structural_adherence",
    "level_calibration",
    "usability",
)

PASS_THRESHOLD = 4.0

JUDGE_SYSTEM_PROMPT = (
    "أنت مصحّح تربوي صارم ومتشكك، لا مقيّم متساهل. تفترض أن المادة المعروضة عليك "
    "قد تحتوي على أخطاء حتى تثبت العكس بالتدقيق الفعلي. لا تُعطِ درجة كاملة إلا "
    "لمادة خالية تمامًا من أي عيب لغوي أو بنيوي أو منهجي."
)

_JUDGE_INSTRUCTION_TEMPLATE = """أنت مصحّح صارم، لا مقيّم متساهل. مهمتك إيجاد العيوب، لا تأكيد الجودة.

الخطوة 1 - تدقيق فعلي، ابحث بدقة عن:
- أخطاء لغوية: جمع غير قياسي، خطأ في الإعراب أو التذكير والتأنيث، خطأ إملائي أو نحوي، حتى لو كان طفيفًا.
- عدم اتساق بنيوي: وقت مقدّر لا يتناسب مع حجم المحتوى (كامتحان من خمس أسئلة بوقت أقل من 10 دقائق)، غياب مفتاح إجابة مطلوب، ترقيم غير واضح.
- انحراف عن المنهاج: أي معلومة أو مفهوم غير وارد في السياق المرفق أدناه، أو تناقض معه.
- عدم ملاءمة للمستوى: صعوبة أو أسلوب لا يناسب المستوى الدراسي المذكور.

اكتب أولًا لائحة بكل عيب وجدته (حتى لو كان طفيفًا جدًّا)، أو صرّح بوضوح أنك لم تجد أي عيب إن كان ذلك صحيحًا فعلًا.

الخطوة 2 - الدرجات، بالاعتماد الحصري على العيوب التي وجدتها في الخطوة 1:
5 = لا يوجد أي عيب على الإطلاق في هذا المعيار (نادر، استثنائي).
4 = عيب طفيف واحد لا يؤثر على الاستخدام.
3 = عيب واضح واحد أو أكثر من عيب طفيف.
2 = عيوب متعددة أو عيب جوهري واحد.
1 = غير قابل للاستخدام في هذا المعيار.

المعايير الخمسة:
1. صحة اللغة (language_correctness)
2. الالتزام بالمنهاج (curriculum_fidelity)
3. الالتزام بالبنية (structural_adherence)
4. الملاءمة للمستوى (level_calibration)
5. قابلية الاستخدام (usability)

السياق المنهاجي الأصلي:
{context}

المادة التعليمية المُنشأة (نوع المهمة: {task_type}):
{output}

أعد الإجابة بصيغة JSON فقط دون أي نص إضافي أو علامات ترميز، وفق هذا المخطط بالضبط:
{{"issues_found": ["<عيب 1>", "<عيب 2>"], "language_correctness": <1-5>, "curriculum_fidelity": <1-5>, "structural_adherence": <1-5>, "level_calibration": <1-5>, "usability": <1-5>, "notes": "<ملاحظة قصيرة بالعربية>"}}
إن لم تجد أي عيب اجعل issues_found مصفوفة فارغة [].
عند الاستشهاد بكلمة أو عبارة داخل issues_found أو notes، استخدم علامتي التنصيص « » ولا تستخدم علامتي التنصيص الإنجليزيتين " " حتى لا يتعارضا مع تنسيق JSON.
"""

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class JudgeResult:
    scores: dict
    average: float
    passed: bool
    notes: str
    issues_found: list


_SCORE_FIELD_RES = {
    k: re.compile(rf'"{k}"\s*:\s*([\d.]+)') for k in RUBRIC_CRITERIA
}


def _regex_fallback_parse(raw: str) -> dict:
    result = {}
    for key, pattern in _SCORE_FIELD_RES.items():
        match = pattern.search(raw)
        if not match:
            raise ValueError(f"could not recover '{key}' score via regex fallback")
        result[key] = float(match.group(1))
    return result


def _parse_judge_response(raw: str) -> dict:
    cleaned = _JSON_FENCE_RE.sub("", raw.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # A quoted word inside issues_found/notes sometimes breaks strict JSON
        # (unescaped " within a " delimited string); the numeric scores
        # usually still survive intact, so recover those directly rather
        # than dropping the whole judgment.
        return _regex_fallback_parse(cleaned)


def build_judge_prompt(context: str, generated_output: str, task_type: str) -> str:
    return _JUDGE_INSTRUCTION_TEMPLATE.format(
        context=context, output=generated_output, task_type=task_type
    )


def judge_pair(pair: dict, client: TeacherModelClient) -> JudgeResult:
    context = pair["messages"][1]["content"]
    generated_output = pair["messages"][2]["content"]
    user_prompt = build_judge_prompt(context, generated_output, pair["task_type"])

    raw = client.generate(JUDGE_SYSTEM_PROMPT, user_prompt)
    parsed = _parse_judge_response(raw)

    scores = {k: float(parsed[k]) for k in RUBRIC_CRITERIA}
    average = sum(scores.values()) / len(scores)
    return JudgeResult(
        scores=scores,
        average=round(average, 2),
        passed=average >= PASS_THRESHOLD,
        notes=parsed.get("notes", ""),
        issues_found=parsed.get("issues_found", []),
    )


def judge_dataset(pairs: list[dict], client: TeacherModelClient) -> tuple[list[dict], list[dict]]:
    judged: list[dict] = []
    failures: list[dict] = []
    for pair in pairs:
        try:
            result = judge_pair(pair, client)
            judged_pair = dict(pair)
            judged_pair["judge"] = {
                "scores": result.scores,
                "average": result.average,
                "passed": result.passed,
                "notes": result.notes,
                "issues_found": result.issues_found,
            }
            judged.append(judged_pair)
        except Exception as e:
            failures.append(
                {
                    "source_entry_id": pair.get("source_entry_id"),
                    "task_type": pair.get("task_type"),
                    "error": str(e),
                }
            )
    return judged, failures
