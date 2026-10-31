"""Deterministic report construction."""

from typing import Any

from agent.schemas import Report, TestResult


def build_report(task: str, records: list[dict[str, Any]]) -> Report:
    results = [TestResult.model_validate(record) for record in records]
    passed = sum(result.status == "PASS" for result in results)
    failed = sum(result.status == "FAIL" for result in results)
    errors = sum(result.status == "ERROR" for result in results)
    if not results or errors == len(results):
        overall = "ERROR"
    elif errors:
        overall = "PARTIAL"
    elif failed:
        overall = "FAIL"
    else:
        overall = "PASS"
    summary = f"{passed} passed, {failed} failed" + (f", {errors} errored." if errors else ".")
    return Report(
        task=task,
        overall_status=overall,
        summary=summary,
        total_tests=len(results),
        passed=passed,
        failed=failed,
        results=results,
        limitations=["Results come from the built-in synthetic demo API."],
    )

