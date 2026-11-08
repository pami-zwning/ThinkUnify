"""Scan multiple parquet dataset directories, generate per-dir summary + total summary.

Usage:
    python scripts/parquet_manifest.py

Edit TASKS and DATA_ROOT below to match your setup.
Each task dir should contain chunk_*.parquet files.

Output:
  - Per task: {data_dir}/{task_name}_parquet_summary.json  (per-file manifest)
  - Total:   {DATA_ROOT}/datasets_summary.json             (aggregated summary)
"""

import os
import json
import glob
import pyarrow.parquet as pq


# ================= Config =================
DATA_ROOT = "/mnt/afs/interleaved_data/ningzhiwei/code/ThinkMorph/data/sft_data"

TASKS = [
    "ThinkMorph_Jigsaw_Assembly",
    "ThinkMorph_Visual_Search",
    "ThinkMorph_Chart_Refocus",
    "ThinkMorph_Spatial_Navigation",
]
# ==========================================


def scan_dir(parquet_dir):
    """Scan one parquet directory, write per-file manifest, return summary."""
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        print(f"  [SKIP] No parquet files in {parquet_dir}")
        return None

    dir_name = os.path.basename(os.path.normpath(parquet_dir))
    manifest = {}
    total_rows = 0

    for f in files:
        pf = pq.ParquetFile(f)
        nr = pf.metadata.num_rows
        nrg = pf.metadata.num_row_groups
        manifest[f] = {"num_rows": nr, "num_row_groups": nrg}
        total_rows += nr

    # Write per-file manifest
    manifest_path = os.path.join(parquet_dir, f"{dir_name}_parquet_summary.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    print(f"  {dir_name}: {len(files)} files, {total_rows} samples -> {manifest_path}")

    return {
        "data_dir": parquet_dir,
        "num_files": len(files),
        "num_total_samples": total_rows,
        "parquet_info_path": manifest_path,
    }


def main():
    print(f"Scanning {len(TASKS)} datasets under {DATA_ROOT}\n")

    total_summary = {}
    for task in TASKS:
        parquet_dir = os.path.join(DATA_ROOT, task)
        summary = scan_dir(parquet_dir)
        if summary:
            total_summary[task] = summary

    # Write total summary
    total_path = os.path.join(DATA_ROOT, "datasets_summary.json")
    with open(total_path, "w", encoding="utf-8") as fp:
        json.dump(total_summary, fp, indent=4, ensure_ascii=False)

    total_files = sum(s["num_files"] for s in total_summary.values())
    total_samples = sum(s["num_total_samples"] for s in total_summary.values())
    print(f"\nTotal: {len(total_summary)} datasets, {total_files} files, {total_samples} samples")
    print(f"Summary saved to: {total_path}")


if __name__ == "__main__":
    main()
