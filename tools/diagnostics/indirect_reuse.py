"""Report the validated indirect-light reuse memory plan."""

import argparse

from ordinarylight.integrations.indirect_reuse import IndirectReservoirPlan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--budget-mib", type=float, default=128.0)
    args = parser.parse_args()
    plan = IndirectReservoirPlan(
        args.width, args.height, scale=args.scale, budget_mib=args.budget_mib)
    print(
        f"indirect reservoirs: {plan.width}x{plan.height} | "
        f"{plan.bytes_per_reservoir} B/pixel | "
        f"{plan.history_frames} frames | {plan.estimated_mib:.1f} MiB")


if __name__ == "__main__":
    main()
