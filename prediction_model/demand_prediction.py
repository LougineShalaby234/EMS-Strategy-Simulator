import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Union, Tuple, Sequence, Optional, overload
from datetime import datetime


class AbstractPredictionModel:
    def __init__(self, weight_path=None):
        """Initialize the prediction model.
        
        Parameters
        ----------
        weight_path
            Path to the model weights.
        """
        pass

    def inference(self, area_space, area_time=None, events=None, resolution=10):
        """Make predictions on the given input.
        
        Parameters
        ----------
        area_space
            Spatial information for prediction
        area_time
            Temporal information for prediction
        events
            Optional event data
        resolution
            Spatial resolution for prediction
            
        Returns
        -------
        Prediction results
        """
        pass


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------
class LightGBMRLPEventPrediction(AbstractPredictionModel):
    """LightGBM‑based event demand prediction.

    Parameters
    ----------
    weight_path
        Path to the *pickled* LightGBM model (.pkl or .joblib).
    resolution
        Spatial grid resolution in degrees used to discretise latitude and
        longitude (default is 10 – i.e. 0.1° × 0.1° cells).
    """
    _FEATURES: Tuple[str, ...] = (
        "hour_sin",
        "hour_cos",
        "dayofweek",
        "season",
        "lat_grid",
        "lon_grid",
        "month",
        "dayofmonth",
    )

    def __init__(self, model_path: Union[str, Path], resolution: int = 10):
        super().__init__(model_path)  # Call parent constructor
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        self._resolution = resolution
        self._two_pi = 2.0 * math.pi

    # ---------------------------------------------------------------------
    # Public API – only tuples of (lat, lon, timestamp) are accepted
    # ---------------------------------------------------------------------
    @overload
    def inference(
        self,
        point: Tuple[float, float, datetime],
        *,
        resolution: Optional[int] = None,
    ) -> np.ndarray: ...

    @overload
    def inference(
        self,
        points: Sequence[Tuple[float, float, datetime]],
        *,
        resolution: Optional[int] = None,
    ) -> np.ndarray: ...

    def inference(
        self,
        area_space: Union[Tuple[float, float, datetime], Sequence[Tuple[float, float, datetime]]],
        area_time=None,
        events=None,
        resolution: Optional[int] = None,
    ) -> np.ndarray:
        """Predict emergency calls for one or multiple (lat, lon, timestamp) tuples.
        
        Parameters
        ----------
        area_space
            Single tuple or sequence of (latitude, longitude, datetime)
        area_time
            Not used in this implementation (kept for compatibility with abstract class)
        events
            Not used in this implementation (kept for compatibility with abstract class)
        resolution
            Optional grid resolution override for this prediction
            
        Returns
        -------
        np.ndarray
            Array of predicted call counts
        """
        # Normalize inputs to list of tuples
        if isinstance(area_space, tuple):
            points = [area_space]
        else:
            points = list(area_space)
            
        if not points:
            raise ValueError("Must provide at least one (lat, lon, timestamp) tuple")
            
        # Build feature DataFrame
        res = resolution if resolution is not None else self._resolution
        df = self._build_feature_frame(points, res)
        
        # Ensure we're only using the required features for prediction
        features_df = df[list(self._FEATURES)]
        
        # Make predictions
        return self.model.predict(features_df)

    def _build_feature_frame(
        self, 
        points: Sequence[Tuple[float, float, datetime]], 
        resolution: int
    ) -> pd.DataFrame:
        """Create feature DataFrame from input points."""
        df = pd.DataFrame(points, columns=["lat", "lon", "timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        
        # Temporal features
        dt = df["timestamp"]
        hour_fraction = dt.dt.hour + dt.dt.minute / 60.0
        df["hour_sin"] = np.sin(hour_fraction / 24.0 * self._two_pi)
        df["hour_cos"] = np.cos(hour_fraction / 24.0 * self._two_pi)
        df["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
        df["month"] = dt.dt.month.astype(np.int8)
        df["dayofmonth"] = dt.dt.day.astype(np.int8)
        df["season"] = self._month_to_season(df["month"]).astype(np.int8)
        
        # Spatial features
        df["lat_grid"] = (df["lat"] * resolution).round(0).astype(np.int32)
        df["lon_grid"] = (df["lon"] * resolution).round(0).astype(np.int32)
        
        # Debug to make sure all features exist
        for feature in self._FEATURES:
            if feature not in df.columns:
                raise KeyError(f"Feature '{feature}' is missing from DataFrame. Available columns: {df.columns.tolist()}")
        
        return df

    @staticmethod
    def _month_to_season(months: pd.Series) -> pd.Series:
        """Map month numbers to meteorological seasons (0=DJF, 1=MAM, 2=JJA, 3=SON)."""
        return ((months % 12) // 3).astype(np.int8)
