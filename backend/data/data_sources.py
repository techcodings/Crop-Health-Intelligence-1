"""
More advanced data access stubs.

In production:
- Connect to Sentinel-2 / Landsat / MODIS imagery providers
- Use Open-Meteo, Open-Elevation, soil & irrigation APIs
- Query USDA Quick Stats
Here: generate synthetic but structured tensors and series.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

@dataclass
class FieldContext:
    field_id: str
    lat: float
    lon: float
    geojson: Optional[Dict[str, Any]] = None


def _seasonal_pattern(t: int, period: int = 12) -> float:
    """Simple sine seasonal pattern."""
    return 0.5 + 0.4 * np.sin(2 * np.pi * t / period)


def load_satellite_patch(field: FieldContext, timesteps: int = 12) -> np.ndarray:
    """
    Return synthetic satellite features with a simple seasonal / stress pattern.
    shape: (T, H, W, C)
    C: [NDVI, EVI, SAVI, MSAVI]
    """
    H, W, C = 64, 64, 4
    base = np.zeros((timesteps, H, W, C), dtype="float32")
    for t in range(timesteps):
        season = _seasonal_pattern(t, period=timesteps)
        noise = 0.05 * np.random.randn(H, W, C).astype("float32")
        ndvi = season + noise[..., 0]
        evi = season * 0.95 + noise[..., 1]
        savi = season * 0.9 + noise[..., 2]
        msavi = season * 0.92 + noise[..., 3]
        base[t, ..., 0] = ndvi
        base[t, ..., 1] = evi
        base[t, ..., 2] = savi
        base[t, ..., 3] = msavi

    # Inject a localized "stress event" at the last timestep
    cx, cy = H // 2, W // 2
    r = H // 6
    y, x = np.ogrid[:H, :W]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    base[-1, mask, 0] *= 0.5  # NDVI drop
    base[-1, mask, 1:] *= 0.7
    base = np.clip(base, 0.0, 1.0)
    return base


def load_weather_series(field: FieldContext, timesteps: int = 12) -> np.ndarray:
    """
    Synthetic weather:
    features: [temp (C), precip (mm), humidity (%), wind (m/s)]
    shape: (T, F)
    """
    t = np.arange(timesteps)
    temp = 25 + 7 * np.sin(2 * np.pi * t / timesteps) + np.random.randn(timesteps)
    precip = np.maximum(0, np.random.gamma(shape=2.0, scale=3.0, size=timesteps))
    humidity = 60 + 15 * np.sin(2 * np.pi * (t + 3) / timesteps) + 5 * np.random.randn(timesteps)
    wind = 2 + np.abs(np.random.randn(timesteps))

    series = np.stack(
        [
            (temp - 10) / 25.0,
            precip / 20.0,
            humidity / 100.0,
            wind / 10.0,
        ],
        axis=1,
    ).astype("float32")
    return series


def load_static_features(field: FieldContext) -> np.ndarray:
    """
    Static per-field features (normalized):
    - elevation, slope, aspect, curvature
    - soil type one-hot (5)
    - irrigation coverage
    - misc land-use features
    """
    elevation = np.random.uniform(0, 1)
    slope = np.random.uniform(0, 1)
    aspect = np.random.uniform(0, 1)
    curvature = np.random.uniform(0, 1)
    soil_type = np.zeros(5, dtype="float32")
    soil_type[np.random.randint(0, 5)] = 1.0
    irrigation = np.random.uniform(0, 1)
    misc = np.random.rand(4).astype("float32")
    vec = np.concatenate([[elevation, slope, aspect, curvature], soil_type, [irrigation], misc])
    return vec.astype("float32")


def load_regional_stats(field: FieldContext) -> np.ndarray:
    """
    Regional agricultural statistics (normalized):
    e.g. historical yields, typical planting/harvest dates, etc.
    """
    return np.random.rand(8).astype("float32")


def load_all_inputs(field: FieldContext, timesteps: int = 12) -> Dict[str, Any]:
    return {
        "satellite": load_satellite_patch(field, timesteps),
        "weather": load_weather_series(field, timesteps),
        "static": load_static_features(field),
        "regional": load_regional_stats(field),
    }
