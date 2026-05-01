from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class CollectionConfig:
    name: str
    timestamp_field: Optional[str] = None
    half_life_days: int = 90


@dataclass
class StoreConfig:
    type: str
    connection: dict = field(default_factory=dict)


@dataclass
class ScanConfig:
    cost_budget: int = 50_000
    nli_batch_size: int = 64
    contradiction_candidates_per_chunk: int = 5
    enable_numeric_detector: bool = True
    enable_version_detector: bool = True


@dataclass
class ClusteringConfig:
    min_cluster_size: str | int = "auto"
    use_gpu: bool = False
    auto_tune_clustering: bool = False


@dataclass
class InferenceConfig:
    nli_model: str = "deberta-v3-base-mnli"
    device: str = "cuda"


@dataclass
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass
class PulseConfig:
    store: StoreConfig
    collections: list[CollectionConfig]
    embedding_model: Optional[str] = None
    fixture_dir: Optional[str] = None
    scan: ScanConfig = field(default_factory=ScanConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


def load_config(path: str | Path) -> PulseConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    store_raw = raw.get("store", {})
    store = StoreConfig(
        type=store_raw.get("type", "fixture"),
        connection=store_raw.get("connection", {}),
    )

    collections = [
        CollectionConfig(
            name=c["name"],
            timestamp_field=c.get("timestamp_field"),
            half_life_days=c.get("half_life_days", 90),
        )
        for c in raw.get("collections", [])
    ]

    scan_raw = raw.get("scan", {})
    scan = ScanConfig(
        cost_budget=scan_raw.get("cost_budget", 50_000),
        nli_batch_size=scan_raw.get("nli_batch_size", 64),
        contradiction_candidates_per_chunk=scan_raw.get("contradiction_candidates_per_chunk", 5),
        enable_numeric_detector=scan_raw.get("enable_numeric_detector", True),
        enable_version_detector=scan_raw.get("enable_version_detector", True),
    )

    clustering_raw = raw.get("clustering", {})
    clustering = ClusteringConfig(
        min_cluster_size=clustering_raw.get("min_cluster_size", "auto"),
        use_gpu=clustering_raw.get("use_gpu", False),
    )

    inference_raw = raw.get("inference", {})
    inference = InferenceConfig(
        nli_model=inference_raw.get("nli_model", "deberta-v3-base-mnli"),
        device=inference_raw.get("device", "cuda"),
    )

    dashboard_raw = raw.get("dashboard", {})
    dashboard = DashboardConfig(
        host=dashboard_raw.get("host", "127.0.0.1"),
        port=dashboard_raw.get("port", 8080),
    )

    return PulseConfig(
        store=store,
        collections=collections,
        embedding_model=raw.get("embedding_model"),
        fixture_dir=raw.get("fixture_dir"),
        scan=scan,
        clustering=clustering,
        inference=inference,
        dashboard=dashboard,
    )
