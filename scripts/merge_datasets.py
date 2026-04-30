"""Merge multiple SEIKO Talos datasets into a single JSON file.

Usage:
    python scripts/merge_datasets.py \
        --inputs dataset/seiko_talos_merged.json \
                 dataset/seiko_talos_0428_5000.json \
                 dataset/seiko_talos_0428_7000.json \
        --output dataset/seiko_talos_0428.json
"""
import argparse
import json
from pathlib import Path


def merge(input_paths, output_path):
    merged_poses = []
    merged_contacts = []
    meta = None
    per_file_counts = []

    for p in input_paths:
        with open(p, "r") as f:
            d = json.load(f)
        if meta is None:
            meta = {
                "end_effectors": d.get("end_effectors"),
                "pose_format": d.get("pose_format"),
            }
        else:
            assert d.get("end_effectors") == meta["end_effectors"], f"end_effectors mismatch in {p}"
            assert d.get("pose_format") == meta["pose_format"], f"pose_format mismatch in {p}"

        n_pairs = d.get("num_pairs", len(d["contacts"]) // 2)
        per_file_counts.append((p, n_pairs))
        merged_poses.extend(d["poses"])
        merged_contacts.extend(d["contacts"])

    total_pairs = sum(n for _, n in per_file_counts)
    assert len(merged_poses) == 2 * total_pairs, (len(merged_poses), total_pairs)
    assert len(merged_contacts) == 2 * total_pairs

    out = {
        "description": "SEIKO Talos multi-contact pose pairs (merged)",
        "end_effectors": meta["end_effectors"],
        "pose_format": meta["pose_format"],
        "num_samples": 2 * total_pairs,
        "num_pairs": total_pairs,
        "source_files": [{"path": str(p), "num_pairs": n} for p, n in per_file_counts],
        "poses": merged_poses,
        "contacts": merged_contacts,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f)

    print(f"Merged {len(input_paths)} files -> {output_path}")
    for p, n in per_file_counts:
        print(f"  {p}: {n} pairs")
    print(f"Total: {total_pairs} pairs ({2 * total_pairs} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    merge(args.inputs, args.output)


if __name__ == "__main__":
    main()
