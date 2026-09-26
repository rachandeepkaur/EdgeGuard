"""
Diagnostic (read-only): which Stage 2 check fires on NORMAL validation windows?
Usage: python -m defender.diagnose
"""
from collections import Counter

from defender.defender import Defender
from part1.pipeline import RoadData


def main():
    defender = Defender.load("models", "v2")
    road = RoadData("data/road")
    n = defender.stage2.training_windows
    normal_top = n / (n + 1)
    top_checks, flagged = Counter(), []
    total = 0
    for name in road.manifest.names("validation"):
        for window in road.windows(name, keep_every=road.keep_every(name, 100)):
            total += 1
            result = defender.stage2.score(window)
            if result.score > normal_top:
                top = max(result.sub_scores, key=result.sub_scores.get)
                top_checks[top] += 1
                flagged.append((result.score, result.evidence))
    print(f"Validation windows: {total}")
    print(f"Stage 2 above normal training range: {len(flagged)}")
    print(f"Which check caused it: {dict(top_checks)}")
    print("Top 5 examples:")
    for score, evidence in sorted(flagged, reverse=True)[:5]:
        print(f"  {score:.6f}  {evidence}")


if __name__ == "__main__":
    main()