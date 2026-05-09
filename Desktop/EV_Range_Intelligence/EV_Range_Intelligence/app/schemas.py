from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DriveStyle(str, Enum):
    city = "city"
    mixed = "mixed"
    highway = "highway"


class Terrain(str, Enum):
    flat = "flat"
    hilly = "hilly"
    mountain = "mountain"


class ScenarioInput(BaseModel):
    distance: float = Field(..., gt=0, le=500, description="Trip distance in km")
    duration: float = Field(..., gt=0, le=720, description="Trip duration in minutes")
    soc_start: float = Field(..., ge=5, le=100, description="Battery state of charge in %")
    ambient_temp: float = Field(..., ge=-30, le=55, description="Ambient temperature in °C")
    cabin_temp: float = Field(22, ge=16, le=30, description="Target cabin temperature in °C")
    drive_style: DriveStyle = Field(default=DriveStyle.mixed)
    terrain: Terrain = Field(default=Terrain.hilly)

    @model_validator(mode="after")
    def check_duration_vs_distance(self):
        avg_speed = self.distance / (self.duration / 60)
        if avg_speed > 160:
            raise ValueError("Distance and duration are inconsistent: average speed is unrealistically high.")
        return self


class RawFeatureInput(BaseModel):
    distance: float = Field(..., gt=0, le=500)
    velocity_mean: float = Field(..., ge=0, le=180)
    velocity_max: float = Field(..., ge=0, le=220)
    velocity_std: float = Field(..., ge=0, le=100)
    pct_stationary: float = Field(..., ge=0, le=1)
    pct_city: float = Field(..., ge=0, le=1)
    pct_highway: float = Field(..., ge=0, le=1)
    pct_braking: float = Field(..., ge=0, le=1)
    pct_accelerating: float = Field(..., ge=0, le=1)
    pct_regen: float = Field(..., ge=0, le=1)
    throttle_mean: float = Field(..., ge=0, le=100)
    torque_mean: float = Field(..., ge=-200, le=200)
    accel_std: float = Field(..., ge=0, le=10)
    soc_start: float = Field(..., ge=0, le=100)
    soc_end: float = Field(..., ge=0, le=100)
    battery_temp_mean: float = Field(..., ge=-30, le=80)
    battery_voltage_mean: float = Field(..., ge=200, le=500)
    battery_current_mean: float = Field(..., ge=-500, le=500)
    power_mean: float = Field(..., ge=-100, le=100)
    power_min: float = Field(..., ge=-150, le=150)
    ambient_temp: float = Field(..., ge=-30, le=55)
    climate_power_mean: float = Field(..., ge=0, le=20)
    heating_active_pct: float = Field(..., ge=0, le=1)
    ac_active_pct: float = Field(..., ge=0, le=1)
    temp_diff_mean: float = Field(..., ge=-30, le=40)
    cabin_temp_mean: float = Field(..., ge=16, le=30)
    elevation_mean: float = Field(..., ge=-500, le=5000)
    elevation_change_std: float = Field(..., ge=0, le=50)
    elevation_gain: float = Field(..., ge=0, le=500)
    soc_drop: float = Field(..., ge=0, le=100)
    duration_min: float = Field(..., gt=0, le=720)

    @model_validator(mode="after")
    def validate_soc(self):
        if self.soc_end > self.soc_start:
            raise ValueError("soc_end cannot be greater than soc_start.")
        total_mode = self.pct_stationary + self.pct_city + self.pct_highway
        if total_mode > 1.35:
            raise ValueError("pct_stationary + pct_city + pct_highway is too large to be realistic.")
        return self


class PredictionResponse(BaseModel):
    prediction: Literal["HIGH CONSUMPTION", "LOW CONSUMPTION"]
    confidence: float
    high_probability: float
    low_probability: float
    estimated_range_km: float
    estimated_consumption_kwh_km: float
    energy_available_kwh: float
    scenario_summary: str
    driving_style: str
    terrain: str
    climate_load_kw: float
    notes: list[str]
