"""Compare captured temporal sequences for formal quality gates."""

import argparse

from ordinarylight.integrations.temporal_quality import (
    load_hdr_sequence,
    summarize_temporal_quality,
    write_temporal_quality_csv,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--conventional", required=True)
    parser.add_argument("--restir", required=True)
    parser.add_argument("--csv", default="/tmp/ordinarylight_temporal_quality.csv")
    args = parser.parse_args()
    reference, reference_metadata = load_hdr_sequence(args.reference)
    conventional, _ = load_hdr_sequence(args.conventional)
    restir, _ = load_hdr_sequence(args.restir)
    comparisons = {
        "conventional": (reference, conventional),
        "restir": (reference, restir),
    }
    print("Reference metadata:", reference_metadata)
    for mode, pair in comparisons.items():
        print(mode)
        for name, value in summarize_temporal_quality(*pair).items():
            print(f"  {name}={value:.8g}")
    write_temporal_quality_csv(args.csv, comparisons)
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
