from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PathStepExpectation(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str = Field(min_length=1)
    via_edge_type: str | None = None

    @field_validator("via_edge_type")
    @classmethod
    def empty_edge_type_is_not_allowed(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("via_edge_type cannot be empty")
        return value


class PathShapeExpectation(BaseModel):
    model_config = ConfigDict(frozen=True)

    match_mode: Literal["subsequence", "prefix", "exact"] = "subsequence"
    steps: list[PathStepExpectation] = Field(min_length=1)


class ActivationTraceStepExpectation(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str = Field(min_length=1)
    edge_type: str | None = None
    hop: int | None = Field(default=None, ge=0)
    is_seed: bool | None = None
    stopped_reason: str | None = None


class ActivationTraceShapeExpectation(BaseModel):
    model_config = ConfigDict(frozen=True)

    match_mode: Literal["subsequence", "prefix", "exact"] = "subsequence"
    steps: list[ActivationTraceStepExpectation] = Field(min_length=1)


class StructuredBenchmarkExpectation(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_node_ids: list[str] = Field(min_length=1)
    minimum_evidence_matches: int = Field(default=1, ge=1)
    path: PathShapeExpectation | None = None
    activation_trace: ActivationTraceShapeExpectation | None = None
    path_scope: Literal["any_path", "best_path"] = "any_path"
    required_edge_types: list[str] = Field(default_factory=list)
    required_semantic_roles: list[str] = Field(default_factory=list)
    required_contradiction_pairs: list[tuple[str, str]] = Field(default_factory=list)
    required_trace_stop_reasons: list[str] = Field(default_factory=list)
    min_activation_trace_length: int | None = Field(default=None, ge=0)
    max_activation_trace_length: int | None = Field(default=None, ge=0)

    @field_validator("minimum_evidence_matches")
    @classmethod
    def minimum_matches_must_not_exceed_expected_evidence(cls, value: int, info):
        evidence_node_ids = info.data.get("evidence_node_ids", [])
        if evidence_node_ids and value > len(evidence_node_ids):
            raise ValueError("minimum_evidence_matches cannot exceed evidence_node_ids length")
        return value

    @model_validator(mode="after")
    def activation_trace_length_bounds_must_be_consistent(self):
        if (
            self.min_activation_trace_length is not None
            and self.max_activation_trace_length is not None
            and self.max_activation_trace_length < self.min_activation_trace_length
        ):
            raise ValueError(
                "max_activation_trace_length cannot be less than min_activation_trace_length"
            )
        return self


class StructuredBenchmarkCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    expectation: StructuredBenchmarkExpectation


class StructuredBenchmarkDataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    domain_pack_name: str = Field(min_length=1)
    document_directory: str = Field(min_length=1)
    cases: list[StructuredBenchmarkCase] = Field(min_length=1)


class StructuredBenchmarkCaseReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    query: str
    tags: list[str] = Field(default_factory=list)
    evidence_hit: bool
    hit: bool
    path_hit: bool | None = None
    activation_trace_hit: bool | None = None
    semantic_hit: bool | None = None
    contradiction_hit: bool | None = None
    expected_evidence: list[str]
    matched_evidence: list[str]
    missing_evidence: list[str]
    returned_node_ids: list[str]
    surfaced_semantic_roles: list[str] = Field(default_factory=list)
    surfaced_contradictions: list[tuple[str, str]] = Field(default_factory=list)
    path_edge_types: list[str] = Field(default_factory=list)
    activated_node_count: int = Field(default=0, ge=0)
    activation_trace_length: int = Field(default=0, ge=0)
    activation_stopped_reasons: list[str] = Field(default_factory=list)
    best_path_hops: int = Field(default=0, ge=0)
    best_path_exception_score: float = Field(default=0.0, ge=0.0)
    best_path_contradiction_score: float = Field(default=0.0, ge=0.0)
    best_answer: str = ""
    latency_ms: float = Field(ge=0.0)


class StructuredBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str
    retriever_name: str
    questions: int = Field(ge=0)
    evidence_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_recall: float = Field(ge=0.0, le=1.0)
    avg_latency_ms: float = Field(ge=0.0)
    case_reports: list[StructuredBenchmarkCaseReport] = Field(default_factory=list)


class StructuredBenchmarkModeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_recall: float = Field(ge=0.0, le=1.0)
    avg_latency_ms: float = Field(ge=0.0)
    path_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    activation_trace_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_activated_nodes: float = Field(default=0.0, ge=0.0)
    avg_propagation_depth: float = Field(default=0.0, ge=0.0)
    avg_activation_trace_length: float = Field(default=0.0, ge=0.0)


class StructuredBenchmarkModeCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_hit: bool
    hit: bool
    path_hit: bool | None = None
    activation_trace_hit: bool | None = None
    semantic_hit: bool | None = None
    contradiction_hit: bool | None = None
    matched_evidence: list[str] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)


class StructuredBenchmarkComparisonCaseReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    modes: dict[str, StructuredBenchmarkModeCaseResult]
    best_modes: list[str] = Field(default_factory=list)
    missed_by_all: bool = False


class StructuredBenchmarkComparisonReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    per_question: list[StructuredBenchmarkComparisonCaseReport] = Field(default_factory=list)
    mode_summary: dict[str, StructuredBenchmarkModeSummary] = Field(default_factory=dict)


class StructuredBenchmarkSuiteReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str
    modes: dict[str, StructuredBenchmarkReport]
    comparison: StructuredBenchmarkComparisonReport
