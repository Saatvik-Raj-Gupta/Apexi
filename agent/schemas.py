"""Pydantic schemas used at every tool boundary."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from services.safety import (
    MAX_QUERY_PARAMS,
    MAX_QUERY_VALUE_LENGTH,
    MAX_REQUIRED_FIELDS,
    MAX_TASK_LENGTH,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RunTaskInput(StrictModel):
    task: str = Field(min_length=1, max_length=MAX_TASK_LENGTH)


class ListEndpointsInput(StrictModel):
    pass


class RunApiTestInput(StrictModel):
    endpoint_id: str = Field(min_length=1, max_length=64)
    method: Literal["GET"] = "GET"
    expected_status: int = Field(ge=100, le=599)
    query_params: dict[str, str] = Field(default_factory=dict, max_length=MAX_QUERY_PARAMS)
    required_json_fields: list[str] = Field(
        default_factory=list, max_length=MAX_REQUIRED_FIELDS
    )

    @field_validator("query_params")
    @classmethod
    def validate_query_params(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if not key or len(key) > MAX_QUERY_VALUE_LENGTH:
                raise ValueError("Query parameter names must be short and non-empty.")
            if len(item) > MAX_QUERY_VALUE_LENGTH:
                raise ValueError("Query parameter values are too long.")
        return value

    @field_validator("required_json_fields")
    @classmethod
    def validate_required_fields(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Required JSON fields must be unique.")
        if any(not item or len(item) > 64 for item in value):
            raise ValueError("Required JSON fields must be short and non-empty.")
        return value


class ValidateResponseInput(StrictModel):
    test_id: str = Field(min_length=1, max_length=64)
    required_json_fields: list[str] = Field(
        default_factory=list, max_length=MAX_REQUIRED_FIELDS
    )


class TestResult(StrictModel):
    endpoint_id: str
    check: str
    status: Literal["PASS", "FAIL", "ERROR"]
    expected: str
    actual: str
    evidence: str = Field(max_length=1000)


class Report(StrictModel):
    task: str = Field(min_length=1, max_length=MAX_TASK_LENGTH)
    overall_status: Literal["PASS", "FAIL", "PARTIAL", "ERROR"]
    summary: str = Field(min_length=1, max_length=1000)
    total_tests: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    results: list[TestResult] = Field(max_length=100)
    limitations: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, value: list[str]) -> list[str]:
        if any(len(item) > 500 for item in value):
            raise ValueError("Limitations are too long.")
        return value

