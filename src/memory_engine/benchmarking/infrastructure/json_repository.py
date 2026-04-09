from __future__ import annotations

import json
from pathlib import Path

from memory_engine.benchmarking.domain.models import StructuredBenchmarkDataset


class JsonStructuredBenchmarkDatasetRepository:
    def load(self, path: Path) -> StructuredBenchmarkDataset:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return StructuredBenchmarkDataset.model_validate(payload)
