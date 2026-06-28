"""Features domain service — build the feature matrix, then select which columns to keep.

Two classes, same shape (construct with inputs, call one verb):

  - `FeatureBuilder(panel, ...).build()`        -> enriched (store, date) feature frame.
  - `FeatureSelector(model, features).select()` -> the feature names worth keeping.

`subset_features(frame, names)` applies a selected name list back onto a feature frame.
"""

from .builder import LAGS, ROLL_WINDOWS, FeatureBuilder, aggregate_reservations
from .selector import FeatureSelector, subset_features

__all__ = [
    "FeatureBuilder",
    "FeatureSelector",
    "subset_features",
    "aggregate_reservations",
    "LAGS",
    "ROLL_WINDOWS",
]
