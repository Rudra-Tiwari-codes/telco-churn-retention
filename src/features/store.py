"""
Feature store abstractions with metadata and versioning hooks.

This module provides lightweight feature store functionality for tracking
feature definitions, versions, and metadata. Designed to be compatible with
future migration to Feast or Tecton.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class FeatureMetadata:
    """Metadata for a single feature."""

    name: str
    dtype: str
    description: str
    owner: str = "data-team"
    version: str = "1.0"
    tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FeatureSetMetadata:
    """Metadata for a set of features."""

    feature_set_name: str
    features: list[FeatureMetadata]
    created_at: str
    version: str
    column_order: list[str]
    transformer_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save metadata to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> FeatureSetMetadata:
        """Load metadata from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class FeatureStore:
    """Lightweight feature store for tracking feature metadata and versions."""

    def __init__(self, metadata_dir: Path) -> None:
        """Initialize feature store.

        Args:
            metadata_dir: Directory to store feature metadata files.
        """
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def register_feature_set(
        self,
        feature_set_name: str,
        df: pd.DataFrame,
        feature_definitions: list[FeatureMetadata] | None = None,
        transformer_config: dict[str, Any] | None = None,
        version: str | None = None,
    ) -> FeatureSetMetadata:
        """Register a feature set with metadata.

        Args:
            feature_set_name: Name of the feature set.
            df: DataFrame containing the features.
            feature_definitions: Optional list of feature metadata. If None, auto-generated.
            transformer_config: Optional configuration used to generate features.
            version: Optional version string. If None, uses timestamp.

        Returns:
            FeatureSetMetadata object.
        """
        if version is None:
            version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

        # Auto-generate feature definitions if not provided
        if feature_definitions is None:
            feature_definitions = []
            for col in df.columns:
                feature_definitions.append(
                    FeatureMetadata(
                        name=col,
                        dtype=str(df[col].dtype),
                        description=f"Feature: {col}",
                    )
                )

        metadata = FeatureSetMetadata(
            feature_set_name=feature_set_name,
            features=feature_definitions,
            created_at=datetime.now(UTC).isoformat(),
            version=version,
            column_order=list(df.columns),
            transformer_config=transformer_config,
        )

        # Save metadata
        metadata_path = self.metadata_dir / f"{feature_set_name}_{version}.json"
        metadata.save(metadata_path)

        return metadata

    def get_feature_set_metadata(
        self, feature_set_name: str, version: str | None = None
    ) -> FeatureSetMetadata | None:
        """Get metadata for a feature set.

        Args:
            feature_set_name: Name of the feature set.
            version: Optional version. If None, returns latest.

        Returns:
            FeatureSetMetadata or None if not found.
        """
        if version:
            metadata_path = self.metadata_dir / f"{feature_set_name}_{version}.json"
            if metadata_path.exists():
                return FeatureSetMetadata.load(metadata_path)
        else:
            # Find latest version
            pattern = f"{feature_set_name}_*.json"
            matches = list(self.metadata_dir.glob(pattern))
            if matches:
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                return FeatureSetMetadata.load(latest)

        return None

    def list_feature_sets(self) -> list[str]:
        """List all registered feature set names."""
        json_files = list(self.metadata_dir.glob("*.json"))
        feature_sets = set()
        for f in json_files:
            # Extract feature set name (before last underscore)
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2:
                feature_sets.add(parts[0])
        return sorted(feature_sets)
