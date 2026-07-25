from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .generate import TeacherModelClient, generate_dataset
from .judge import judge_dataset
from .pdf_extractor import extract_pdf
from .schema_adapter import SFTPromptCandidate, to_sft_candidates


@dataclass
class ExtractionReport:
    pdf_path: str
    doc_id: str
    total_pages: int
    usable_pages: int
    failed_pages: int
    hard_failure_counts: dict
    soft_warning_counts: dict
    avg_confidence: float
    candidate_count: int
    candidate_counts_by_task_type: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _count_reasons(entries: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for e in entries:
        for reason in e["_extraction_quality"][key]:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def run_extraction(pdf_path: str, doc_id: str | None = None, grade_level: int | None = None) -> tuple[list[dict], list[SFTPromptCandidate], ExtractionReport]:
    entries = extract_pdf(pdf_path, doc_id=doc_id)
    resolved_doc_id = entries[0]["DocId"] if entries else (doc_id or "")

    usable = [e for e in entries if e["_extraction_quality"]["passed"]]
    failed = [e for e in entries if not e["_extraction_quality"]["passed"]]
    confidences = [e["_extraction_quality"]["confidence"] for e in entries]

    candidates: list[SFTPromptCandidate] = []
    for e in usable:
        candidates.extend(to_sft_candidates(e, grade_level=grade_level))

    candidate_counts_by_task_type: dict[str, int] = {}
    for c in candidates:
        candidate_counts_by_task_type[c.task_type.value] = (
            candidate_counts_by_task_type.get(c.task_type.value, 0) + 1
        )

    report = ExtractionReport(
        pdf_path=pdf_path,
        doc_id=resolved_doc_id,
        total_pages=len(entries),
        usable_pages=len(usable),
        failed_pages=len(failed),
        hard_failure_counts=_count_reasons(failed, "hard_failures"),
        soft_warning_counts=_count_reasons(entries, "soft_warnings"),
        avg_confidence=round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        candidate_count=len(candidates),
        candidate_counts_by_task_type=candidate_counts_by_task_type,
    )
    return entries, candidates, report


@dataclass
class DatasetGenerationReport:
    candidates_submitted: int
    pairs_generated: int
    pairs_failed: int
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def run_dataset_generation(
    candidates: list[SFTPromptCandidate],
    client: TeacherModelClient,
    limit: int | None = None,
) -> tuple[list[dict], list[dict], DatasetGenerationReport]:
    submitted = candidates[:limit] if limit is not None else candidates
    results, failures = generate_dataset(submitted, client)
    report = DatasetGenerationReport(
        candidates_submitted=len(submitted),
        pairs_generated=len(results),
        pairs_failed=len(failures),
    )
    return results, failures, report


@dataclass
class JudgeStageReport:
    pairs_submitted: int
    pairs_judged: int
    pairs_accepted: int
    pairs_rejected: int
    pairs_failed: int
    acceptance_rate: float
    judged_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def run_judge_stage(
    pairs: list[dict],
    client: TeacherModelClient,
    limit: int | None = None,
) -> tuple[list[dict], list[dict], JudgeStageReport]:
    submitted = pairs[:limit] if limit is not None else pairs
    judged, failures = judge_dataset(submitted, client)
    accepted = [p for p in judged if p["judge"]["passed"]]
    rejected = [p for p in judged if not p["judge"]["passed"]]

    report = JudgeStageReport(
        pairs_submitted=len(submitted),
        pairs_judged=len(judged),
        pairs_accepted=len(accepted),
        pairs_rejected=len(rejected),
        pairs_failed=len(failures),
        acceptance_rate=round(len(accepted) / len(submitted), 3) if submitted else 0.0,
    )
    return judged, failures, report


def write_jsonl(pairs: list[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def candidate_to_dict(candidate: SFTPromptCandidate) -> dict:
    d = asdict(candidate)
    d["task_type"] = candidate.task_type.value
    return d
