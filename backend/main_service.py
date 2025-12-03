"""
High-level orchestration engine for Crop Health Intelligence (advanced demo).

Responsibilities:
- Load multimodal data for a field
- Encode with Perceiver-style transformer
- Generate stress heatmap (UNet-like)
- Forecast future vegetation index trend
- Use a policy network to recommend actions
- Segment field into management zones via k-means
- Infer likely stress causes (water / nutrient / disease) via simple rules
"""

from typing import Dict, Any, List
import numpy as np
import torch
from sklearn.cluster import KMeans

from backend.data.data_sources import FieldContext, load_all_inputs
from backend.models import (
    SimplePerceiverIO,
    StressMapGenerator,
    SimpleIrrigationPolicy,
    SimpleTemporalForecaster,
)
from backend.utils.config import DEVICE


class CropHealthEngine:
    def __init__(self, timesteps: int = 12, forecast_horizon: int = 6, num_zones: int = 4):
        self.timesteps = timesteps
        self.forecast_horizon = forecast_horizon
        self.num_zones = num_zones

        # Feature dims from data_sources:
        # satellite token: 4 (indices)
        # weather token: 4
        # static: 16
        # regional: 8
        input_dim = 4 + 4 + 14 + 8

        self.encoder = SimplePerceiverIO(input_dim=input_dim).to(DEVICE)
        self.stress_gen = StressMapGenerator(in_channels=4).to(DEVICE)
        self.forecaster = SimpleTemporalForecaster(input_dim=4, horizon=self.forecast_horizon).to(DEVICE)
        self.policy = SimpleIrrigationPolicy(state_dim=256, num_actions=6).to(DEVICE)

        self.encoder.eval()
        self.stress_gen.eval()
        self.forecaster.eval()
        self.policy.eval()

    # ---------- internal helpers ----------

    def _prepare_tokens(self, inputs: Dict[str, Any]) -> torch.Tensor:
        sat = inputs["satellite"]  # (T, H, W, C=4)
        weather = inputs["weather"]  # (T, 4)
        static = inputs["static"]  # (16,)
        regional = inputs["regional"]  # (8,)

        T, H, W, C = sat.shape
        sat_flat = sat.reshape(T, H * W, C)  # (T, HW, 4)

        tokens = []
        for t in range(T):
            s_t = sat_flat[t]  # (HW, 4)
            w_t = np.repeat(weather[t][None, :], H * W, axis=0)  # (HW, 4)
            st = np.repeat(static[None, :], H * W, axis=0)  # (HW, 16)
            reg = np.repeat(regional[None, :], H * W, axis=0)  # (HW, 8)
            tok = np.concatenate([s_t, w_t, st, reg], axis=1)  # (HW, D)
            tokens.append(tok)

        tokens = np.stack(tokens, axis=0)  # (T, HW, D)
        tokens = tokens.reshape(1, -1, tokens.shape[-1])  # (1, T*HW, D)

        return torch.from_numpy(tokens).float().to(DEVICE)

    def _prepare_stress_input(self, inputs: Dict[str, Any]) -> torch.Tensor:
        sat = inputs["satellite"]  # (T, H, W, C)
        last = sat[-1]  # (H, W, C)
        img = np.transpose(last, (2, 0, 1))  # (C, H, W)
        img = img[None, ...]  # (1, C, H, W)
        return torch.from_numpy(img).float().to(DEVICE)

    def _prepare_forecast_input(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        For temporal forecasting we use NDVI/EVI/SAVI/MSAVI means per timestep
        as a compact vegetation time series.
        """
        sat = inputs["satellite"]  # (T, H, W, C)
        T = sat.shape[0]
        series = []
        for t in range(T):
            frame = sat[t]  # (H, W, C)
            mean_vals = frame.reshape(-1, frame.shape[-1]).mean(axis=0)
            series.append(mean_vals)
        series = np.stack(series, axis=0)  # (T, 4)
        return torch.from_numpy(series[None, ...]).float().to(DEVICE)

    def _segment_zones(self, stress_map: np.ndarray, num_zones: int) -> Dict[str, Any]:
        """
        Cluster pixels into management zones based on stress value and position.
        Returns zone id per pixel and summary stats.
        """
        H, W = stress_map.shape
        ys, xs = np.mgrid[0:H, 0:W]
        features = np.stack(
            [
                xs.flatten() / W,
                ys.flatten() / H,
                stress_map.flatten(),
            ],
            axis=1,
        )

        kmeans = KMeans(n_clusters=num_zones, n_init=5, random_state=42)
        labels = kmeans.fit_predict(features)
        labels_img = labels.reshape(H, W)

        zone_stats = []
        for z in range(num_zones):
            mask = labels_img == z
            if mask.sum() == 0:
                mean_stress = float("nan")
            else:
                mean_stress = float(stress_map[mask].mean())
            zone_stats.append(
                {
                    "zone_id": int(z),
                    "mean_stress": mean_stress,
                    "area_fraction": float(mask.mean()),
                }
            )

        return {
            "labels": labels_img.tolist(),
            "zone_stats": zone_stats,
        }

    def _infer_stress_causes(self, inputs: Dict[str, Any], stress_map: np.ndarray) -> List[str]:
        """
        Simple heuristic rules to label potential stress drivers.
        """
        weather = inputs["weather"]  # (T, 4)
        recent_temp = weather[-3:, 0].mean()  # normalized
        recent_precip = weather[-3:, 1].mean()
        recent_humidity = weather[-3:, 2].mean()

        mean_stress = float(stress_map.mean())
        explanations = []

        if mean_stress < 0.4:
            explanations.append("Overall vegetation appears healthy with limited visible stress.")
            return explanations

        # Map normalized values back to approximate physical intuition
        temp_c = recent_temp * 25 + 10
        precip_mm = recent_precip * 20
        humidity_pct = recent_humidity * 100

        if precip_mm < 3 and temp_c > 30:
            explanations.append(
                "Low recent rainfall combined with high temperature suggests potential **water stress**."
            )
        if precip_mm > 10 and humidity_pct > 70:
            explanations.append(
                "High rainfall and humidity may create conditions for **disease pressure** (fungal or bacterial)."
            )
        static = inputs["static"]
        irrigation = float(static[4 + 5])  # after [elev,slope,aspect,curvature] + soil one-hot
        if irrigation < 0.3 and mean_stress > 0.6:
            explanations.append("Limited irrigation coverage with elevated stress hints at **under-irrigation**.")
        if irrigation > 0.7 and mean_stress > 0.6 and precip_mm > 5:
            explanations.append("High irrigation and rainfall with high stress may indicate **waterlogging**.")

        if not explanations:
            explanations.append(
                "Stress pattern is ambiguous from remote signals alone; field scouting is recommended to confirm nutrient or pest issues."
            )
        return explanations

    # ---------- public API ----------

    def run_field(self, field_id: str, lat: float, lon: float) -> Dict[str, Any]:
        field = FieldContext(field_id=field_id, lat=lat, lon=lon)
        inputs = load_all_inputs(field, timesteps=self.timesteps)

        # Encode multimodal tokens
        tokens = self._prepare_tokens(inputs)
        with torch.no_grad():
            enc = self.encoder(tokens)  # (1, 256)

        # Stress map generation
        stress_in = self._prepare_stress_input(inputs)
        with torch.no_grad():
            stress_map = self.stress_gen(stress_in)  # (1,1,H,W)
        stress_np = stress_map.detach().cpu().numpy()[0, 0]

        # Forecast future vegetation index
        forecast_in = self._prepare_forecast_input(inputs)
        with torch.no_grad():
            forecast = self.forecaster(forecast_in)  # (1, horizon)
        forecast_np = forecast.detach().cpu().numpy()[0]

        # Policy recommendation
        with torch.no_grad():
            policy_logits, value = self.policy(enc)  # (1, A), (1,1)
            probs = torch.softmax(policy_logits, dim=-1).detach().cpu().numpy()[0]
            value_scalar = float(value.detach().cpu().numpy()[0, 0])

        actions = [
            "Maintain current irrigation & fertilization regime.",
            "Increase irrigation slightly in high-stress zones.",
            "Decrease irrigation slightly to avoid waterlogging.",
            "Apply nitrogen-focused fertilization to low-vigor areas.",
            "Apply targeted pest/disease control in critical zones.",
            "Schedule in-field scouting to validate remote sensing signals.",
        ]
        best_action_idx = int(probs.argmax())
        best_action = actions[best_action_idx]

        # Alerts
        mean_stress = float(stress_np.mean())
        max_stress = float(stress_np.max())
        alerts: List[str] = []
        if max_stress > 0.85:
            alerts.append("Critical stress detected in localized hotspots.")
        if mean_stress > 0.6:
            alerts.append("Overall field stress elevated; inspect for water or nutrient issues.")
        if mean_stress < 0.3:
            alerts.append("Field appears generally healthy.")
        if not alerts:
            alerts.append("Mild stress detected; continue monitoring.")

        # Management zones
        zones = self._segment_zones(stress_np, num_zones=self.num_zones)

        # Stress cause explanations
        causes = self._infer_stress_causes(inputs, stress_np)

        result = {
            "encoded_state": enc.detach().cpu().numpy()[0].tolist(),
            "stress_map": stress_np.tolist(),
            "forecast_indices": forecast_np.tolist(),
            "policy_probs": probs.tolist(),
            "policy_value": value_scalar,
            "best_action": best_action,
            "alerts": alerts,
            "zones": zones,
            "stress_causes": causes,
            "summary": {
                "mean_stress": mean_stress,
                "max_stress": max_stress,
                "health_score": float(1.0 - mean_stress),
            },
        }
        return result


if __name__ == "__main__":
    engine = CropHealthEngine()
    out = engine.run_field("demo-field", 10.0, 77.0)
    print("Best action:", out["best_action"])
    print("Alerts:", out["alerts"])
    print("Stress causes:", out["stress_causes"])
