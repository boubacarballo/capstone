from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, Union

from vi import Config, Matrix

# str-or-list[str] placeholder so the same field is a single value on the Config
# and a sweepable list on the Matrix (mirrors vi's Mono/Poly trick for int/float).
MatrixStr = TypeVar("MatrixStr", str, list[str])


@dataclass
class ExperimentSchema(Generic[MatrixStr]):
    """Custom experiment fields shared between ExperimentConfig and ExperimentMatrix."""

    experiment: Union[str, MatrixStr] = "baseline"
    learning_mode: Union[str, MatrixStr] = "self"


@dataclass
class ExperimentConfig(Config, ExperimentSchema[str]):
    """A single Config carrying our experiment fields."""


@dataclass
class ExperimentMatrix(Matrix, ExperimentSchema[list[str]]):
    """A Matrix whose experiment fields may be lists, to sweep unique configs."""
