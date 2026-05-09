from app.schemas import ScenarioInput
from app.services.feature_builder import build_features_from_scenario


def test_build_features_from_winter_highway_scenario():
    scenario = ScenarioInput(
        distance=50,
        duration=45,
        soc_start=80,
        ambient_temp=0,
        cabin_temp=22,
        drive_style="highway",
        terrain="hilly",
    )
    context = build_features_from_scenario(scenario)

    assert context.raw_payload["distance"] == 50
    assert context.raw_payload["heating_active_pct"] > 0
    assert context.raw_payload["climate_power_mean"] > 0
    assert context.raw_payload["soc_end"] < 80
    assert context.drive_style == "highway"
