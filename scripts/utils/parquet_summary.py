"""Scan parquet directories and generate a summary JSON.

Usage:
    # Scan a single directory
    python scripts/parquet_summary.py /path/to/parquet_dir

    # Scan all subdirectories under a root (each subdir = one dataset)
    python scripts/parquet_summary.py /path/to/root --recursive

    # Specify output path
    python scripts/parquet_summary.py /path/to/root --recursive -o summary.json
"""

import os
import sys
import json
import glob
import pyarrow.parquet as pq


def scan_one_dir(parquet_dir):
    """Scan a single directory of parquet files, return summary dict + per-file manifest."""
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        return None, None

    manifest = {}
    total_rows = 0
    for f in files:
        pf = pq.ParquetFile(f)
        nr = pf.metadata.num_rows
        nrg = pf.metadata.num_row_groups
        manifest[f] = {"num_rows": nr, "num_row_groups": nrg}
        total_rows += nr

    dataset_name = os.path.basename(parquet_dir)

    # Write per-file manifest
    manifest_path = os.path.join(parquet_dir, f"{dataset_name}_parquet_summary.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    summary = {
        "data_dir": parquet_dir,
        "num_files": len(files),
        "num_total_samples": total_rows,
        "parquet_info_path": manifest_path,
    }
    return dataset_name, summary


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    recursive = "--recursive" in sys.argv or "-r" in sys.argv
    out_path = None
    if "-o" in sys.argv:
        out_path = sys.argv[sys.argv.index("-o") + 1]

    all_summaries = {}

    if recursive:
        # Each immediate subdirectory is treated as one dataset
        subdirs = sorted([
            os.path.join(target, d) for d in os.listdir(target)
            if os.path.isdir(os.path.join(target, d))
        ])
        print(f"Scanning {len(subdirs)} subdirectories under {target}...")
        for sd in subdirs:
            name, summary = scan_one_dir(sd)
            if summary:
                all_summaries[name] = summary
                print(f"  {name}: {summary['num_files']} files, {summary['num_total_samples']} samples")
    else:
        # Single directory
        name, summary = scan_one_dir(target)
        if summary:
            all_summaries[name] = summary
            print(f"  {name}: {summary['num_files']} files, {summary['num_total_samples']} samples")

    if not all_summaries:
        print("No parquet files found.")
        return

    if out_path is None:
        if recursive:
            out_path = os.path.join(target, "datasets_summary.json")
        else:
            out_path = os.path.join(target, f"{os.path.basename(target)}_summary.json")

    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(all_summaries, fp, indent=4, ensure_ascii=False)

    print(f"\nTotal: {len(all_summaries)} datasets, {sum(s['num_total_samples'] for s in all_summaries.values())} samples")
    print(f"Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
