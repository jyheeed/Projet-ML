import os
import re
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("EV_API_URL", "http://localhost:8000").rstrip("/")
PREDICT_ENDPOINT = f"{API_URL}/api/v1/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"

st.set_page_config(page_title="EV Range Assistant", page_icon="⚡", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []


def extract_number(text: str, patterns: list[str], default: float) -> float:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).replace(",", ".")
            return float(val)
    return default


def parse_message(text: str) -> dict[str, Any]:
    text_lower = text.lower()

    distance = extract_number(
        text,
        [
            r"(\d+[\.,]?\d*)\s*(?:km|kilometer|kilometre|kilom[eè]tre)",
            r"(?:distance|trajet|trip|drive|route)\s*(?:de|of|:)?\s*(\d+[\.,]?\d*)",
        ],
        20,
    )

    ambient_temp = extract_number(
        text,
        [
            r"(-?\d+[\.,]?\d*)\s*(?:c|°c|celsius|degr[eé]s?)",
            r"(?:temp|temperature|m[eé]t[eé]o|weather|il fait|it is)\s*(?:de|:)?\s*(-?\d+[\.,]?\d*)",
            r"minus\s*(\d+[\.,]?\d*)",
        ],
        10,
    )
    if any(token in text_lower for token in ["minus", "moins", "negative", "negatif", "négatif"]):
        ambient_temp = -abs(ambient_temp)

    soc_start = extract_number(
        text,
        [
            r"(\d+[\.,]?\d*)\s*(?:%|percent|pour ?cent|pourcent)",
            r"(?:battery|batterie|charge|soc)\s*(?:de|of|at|a|:)?\s*(\d+[\.,]?\d*)",
        ],
        80,
    )

    duration = extract_number(
        text,
        [
            r"(\d+[\.,]?\d*)\s*(?:min|mins|minute|minutes)",
            r"(?:duration|dur[eé]e)\s*(?:de|of|:)?\s*(\d+[\.,]?\d*)",
        ],
        max(15, distance / 45 * 60),
    )

    cabin_temp = extract_number(
        text,
        [r"(?:cabin|habitacle|inside|interior)\s*(?:de|of|at|a|:)?\s*(\d+[\.,]?\d*)"],
        22,
    )

    drive_style = "mixed"
    if any(w in text_lower for w in ["city", "ville", "urbain", "urban", "stop and go", "stop-and-go"]):
        drive_style = "city"
    elif any(w in text_lower for w in ["highway", "autoroute", "motorway", "fast", "rapide"]):
        drive_style = "highway"

    terrain = "hilly"
    if any(w in text_lower for w in ["flat", "plat", "plain", "coast"]):
        terrain = "flat"
    elif any(w in text_lower for w in ["mountain", "montagne", "steep", "alpine"]):
        terrain = "mountain"

    return {
        "distance": round(distance, 1),
        "duration": round(duration, 1),
        "soc_start": round(soc_start, 1),
        "ambient_temp": round(ambient_temp, 1),
        "cabin_temp": round(cabin_temp, 1),
        "drive_style": drive_style,
        "terrain": terrain,
    }


def generate_response(params: dict[str, Any], prediction: dict[str, Any]) -> str:
    confidence_pct = prediction["confidence"] * 100
    remaining_margin = prediction["estimated_range_km"] - params["distance"]
    will_arrive = remaining_margin >= 0

    lines = [
        "### Trip analysis",
        "",
        f"- Distance: **{params['distance']} km**",
        f"- Ambient temperature: **{params['ambient_temp']}°C**",
        f"- Start battery: **{params['soc_start']}%**",
        f"- Drive style: **{params['drive_style']}**",
        f"- Terrain: **{params['terrain']}**",
        "",
        f"**Prediction:** {prediction['prediction']} ({confidence_pct:.0f}% confidence)",
        f"- Estimated range: **{prediction['estimated_range_km']} km**",
        f"- Estimated consumption: **{prediction['estimated_consumption_kwh_km']} kWh/km**",
        f"- Energy available: **{prediction['energy_available_kwh']} kWh**",
        "",
    ]

    if will_arrive:
        lines.append(f"You should complete the trip with about **{remaining_margin:.0f} km** of range margin.")
    else:
        lines.append(f"The trip looks risky. You are short by about **{abs(remaining_margin):.0f} km** of range.")

    if prediction.get("notes"):
        lines.append("")
        lines.append("**Practical notes**")
        for note in prediction["notes"][:3]:
            lines.append(f"- {note}")

    return "\n".join(lines)


with st.sidebar:
    st.title("⚡ EV Range Assistant")
    st.markdown("BMW i3 trip estimator based on the trained ML model")
    st.divider()
    st.markdown("**Example inputs**")
    st.markdown("- `I need to drive 40 km at -3C with 60% battery on highway.`")
    st.markdown("- `Trip of 18 km, 75 percent battery, city driving, flat terrain.`")
    st.divider()
    try:
        health = requests.get(HEALTH_ENDPOINT, timeout=3).json()
        st.success(f"API connected · v{health['version']}")
    except requests.RequestException:
        st.error("API unreachable. Start the FastAPI service first.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("⚡ EV Range Assistant")
st.caption("Ask about a BMW i3 trip in simple English or French.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Example: I need to drive 40 km, it is -3C, I have 60% battery"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running prediction..."):
            params = parse_message(prompt)
            try:
                response = requests.post(PREDICT_ENDPOINT, json=params, timeout=8)
                response.raise_for_status()
                prediction = response.json()
                message = generate_response(params, prediction)
            except requests.HTTPError:
                detail = "Invalid request."
                try:
                    detail = response.json().get("detail", detail)
                except Exception:
                    pass
                message = f"API rejected the request: {detail}"
            except requests.RequestException:
                message = "Unable to reach the API. Start the FastAPI server on port 8000 or set EV_API_URL."
        st.markdown(message)

    st.session_state.messages.append({"role": "assistant", "content": message})
