"""
MOCK traffic generator. NOT REAL DATA.

Used only by nano_runner.py --mock, so the Nano run and the latency
measurement can be tested before Part 1's real windows exist.
Results produced with this traffic must be labelled MOCK and must never
be reported as detection results on ROAD.
"""

from shared.schemas import TrafficWindow

# MOCK periodic IDs: ID -> period in seconds (invented, not from ROAD)
MOCK_PERIODS = {"0D0": 0.010, "0F4": 0.020, "1A0": 0.050}


def make_mock_normal_window(index: int, capture_id: str = "cap90",
                            window_s: float = 1.0) -> TrafficWindow:
    start = 1080000000.0 + index * window_s
    frames = []
    for can_id, period in MOCK_PERIODS.items():
        for k in range(int(round(window_s / period))):
            jitter = (((index * 7 + k * 3) % 5) - 2) * 0.0002
            t = round(start + 0.001 + k * period + jitter, 6)
            if start <= t <= start + window_s:
                frames.append({"timestamp": t, "can_id": can_id,
                               "payload": "0011223344556677"})
    frames.sort(key=lambda f: f["timestamp"])
    return TrafficWindow(window_id=f"{capture_id}_w{index:04d}", capture_id=capture_id,
                         window_start=start, window_end=start + window_s, frames=frames)


def make_mock_fuzzing_window(index: int, capture_id: str = "cap91") -> TrafficWindow:
    """MOCK attack: normal window plus 10 frames from unknown ID 7FF."""
    data = make_mock_normal_window(index, capture_id).model_dump()
    start = data["window_start"]
    data["frames"] += [{"timestamp": round(start + 0.1 + k * 0.05, 6), "can_id": "7FF",
                        "payload": "FFFFFFFFFFFFFFFF"} for k in range(10)]
    data["frames"].sort(key=lambda f: f["timestamp"])
    return TrafficWindow(**data)