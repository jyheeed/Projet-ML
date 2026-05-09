from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.config import HIGH_CONSUMPTION_RATE, LOW_CONSUMPTION_RATE
from app.schemas import DriveStyle, ScenarioInput, Terrain


@dataclass(frozen=True)
class ScenarioContext:
    drive_style: str
    terrain: str
    climate_power_kw: float
    temperature_gap: float
    energy_used_kwh: float
    soc_end: float
    raw_payload: dict[str, float]


DRIVE_PROFILES: dict[str, dict[str, float]] = {
    "city": {
        "vel_mean": 28, "vel_max": 55, "vel_std": 18,
        "pct_stat": 0.30, "pct_city": 0.60, "pct_hwy": 0.02,
        "pct_brake": 0.15, "pct_accel": 0.18, "throttle": 15,
        "torque": 12, "accel_std": 1.5,
    },
    "mixed": {
        "vel_mean": 42, "vel_max": 90, "vel_std": 25,
        "pct_stat": 0.18, "pct_city": 0.40, "pct_hwy": 0.15,
        "pct_brake": 0.12, "pct_accel": 0.14, "throttle": 22,
        "torque": 18, "accel_std": 1.2,
    },
    "highway": {
        "vel_mean": 85, "vel_max": 130, "vel_std": 20,
        "pct_stat": 0.05, "pct_city": 0.10, "pct_hwy": 0.65,
        "pct_brake": 0.06, "pct_accel": 0.08, "throttle": 35,
        "torque": 28, "accel_std": 0.7,
    },
}

TERRAIN_PROFILES: dict[str, dict[str, float]] = {
    "flat": {"elev": 480, "elev_std": 0.05, "elev_gain": 2},
    "hilly": {"elev": 530, "elev_std": 0.3, "elev_gain": 15},
    "mountain": {"elev": 600, "elev_std": 0.8, "elev_gain": 45},
}

REGEN_BY_STYLE = {"city": 0.12, "mixed": 0.08, "highway": 0.04}


def build_features_from_scenario(scenario: ScenarioInput) -> ScenarioContext:
    drive_style = scenario.drive_style.value if isinstance(scenario.drive_style, DriveStyle) else str(scenario.drive_style)
    terrain = scenario.terrain.value if isinstance(scenario.terrain, Terrain) else str(scenario.terrain)

    p = DRIVE_PROFILES[drive_style]
    t = TERRAIN_PROFILES[terrain]

    temp_diff = scenario.cabin_temp - scenario.ambient_temp
    needs_heating = scenario.ambient_temp < 15
    needs_ac = scenario.ambient_temp > 25

    heating_pct = min(1.0, max(0.0, (15 - scenario.ambient_temp) / 20)) if needs_heating else 0.0
    ac_pct = min(1.0, max(0.0, (scenario.ambient_temp - 25) / 15)) if needs_ac else 0.0
    climate_power = abs(temp_diff) * 0.25 if needs_heating else (abs(temp_diff) * 0.15 if needs_ac else 0.1)

    battery_temp = scenario.ambient_temp + 3 if scenario.ambient_temp < 5 else scenario.ambient_temp + 1
    voltage = 365 + battery_temp * 0.5
    power_mean = -(p["vel_mean"] * 0.18 + climate_power)
    current_mean = power_mean / (voltage / 1000)
    power_min = -(p["vel_max"] * 0.35 + climate_power)

    consumption_rate = HIGH_CONSUMPTION_RATE if needs_heating else LOW_CONSUMPTION_RATE
    energy_used = scenario.distance * consumption_rate
    soc_drop = (energy_used / 18.8) * 100
    soc_end = max(0.0, scenario.soc_start - soc_drop)

    payload = {
        "distance": float(scenario.distance),
        "velocity_mean": p["vel_mean"],
        "velocity_max": p["vel_max"],
        "velocity_std": p["vel_std"],
        "pct_stationary": p["pct_stat"],
        "pct_city": p["pct_city"],
        "pct_highway": p["pct_hwy"],
        "pct_braking": p["pct_brake"],
        "pct_accelerating": p["pct_accel"],
        "pct_regen": REGEN_BY_STYLE[drive_style],
        "throttle_mean": p["throttle"],
        "torque_mean": p["torque"],
        "accel_std": p["accel_std"],
        "soc_start": float(scenario.soc_start),
        "soc_end": float(soc_end),
        "battery_temp_mean": float(battery_temp),
        "battery_voltage_mean": float(voltage),
        "battery_current_mean": float(current_mean),
        "power_mean": float(power_mean),
        "power_min": float(power_min),
        "ambient_temp": float(scenario.ambient_temp),
        "climate_power_mean": float(climate_power),
        "heating_active_pct": float(heating_pct),
        "ac_active_pct": float(ac_pct),
        "temp_diff_mean": float(temp_diff),
        "cabin_temp_mean": float(scenario.cabin_temp),
        "elevation_mean": t["elev"],
        "elevation_change_std": t["elev_std"],
        "elevation_gain": t["elev_gain"],
        "soc_drop": float(soc_drop),
        "duration_min": float(scenario.duration),
    }
    return ScenarioContext(
        drive_style=drive_style,
        terrain=terrain,
        climate_power_kw=float(climate_power),
        temperature_gap=float(temp_diff),
        energy_used_kwh=float(energy_used),
        soc_end=float(soc_end),
        raw_payload=payload,
    )


def payload_to_model_dataframe(payload: dict[str, float], feature_names: list[str]) -> pd.DataFrame:
    column_mapping = {
        "distance": "Distance",
        "velocity_mean": "Velocity_mean",
        "velocity_max": "Velocity_max",
        "velocity_std": "Velocity_std",
        "pct_stationary": "Pct_Stationary",
        "pct_city": "Pct_City",
        "pct_highway": "Pct_Highway",
        "pct_braking": "Pct_Braking",
        "pct_accelerating": "Pct_Accelerating",
        "pct_regen": "Pct_Regen",
        "throttle_mean": "Throttle_mean",
        "torque_mean": "Torque_mean",
        "accel_std": "Accel_std",
        "soc_start": "SoC_start",
        "soc_end": "SoC_end",
        "battery_temp_mean": "Battery_Temp_mean",
        "battery_voltage_mean": "Battery_Voltage_mean",
        "battery_current_mean": "Battery_Current_mean",
        "power_mean": "Power_mean",
        "power_min": "Power_min",
        "ambient_temp": "Ambient_Temp",
        "climate_power_mean": "Climate_Power_mean",
        "heating_active_pct": "Heating_Active_Pct",
        "ac_active_pct": "AC_Active_Pct",
        "temp_diff_mean": "Temp_Diff_mean",
        "cabin_temp_mean": "Cabin_Temp_mean",
        "elevation_mean": "Elevation_mean",
        "elevation_change_std": "Elevation_Change_std",
        "elevation_gain": "Elevation_gain",
        "soc_drop": "SoC_Drop",
        "duration_min": "Duration_min",
    }
    mapped = {column_mapping[key]: value for key, value in payload.items()}
    return pd.DataFrame([mapped])[feature_names]
