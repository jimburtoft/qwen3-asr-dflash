"""Estimate VoxPopuli en hours per split from parquet footer metadata only (no audio download).
Audio is 16 kHz mono float32 WAV (4 bytes/sample, 44-byte header), stored uncompressed in the
parquet `audio.bytes` column, so hours ~= (uncompressed bytes - 44 * rows) / 64000 / 3600.
Usage: python voxpopuli_hours.py
"""
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO, REV = "facebook/voxpopuli", "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
fs = HfFileSystem()
fl = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REV)
for pref in ("en/train-", "en/validation-", "en/test-", "en_accented/test-"):
    rows = by = 0
    for f in sorted(x for x in fl if x.startswith(pref)):
        with fs.open(f"datasets/{REPO}@{REV}/{f}") as fh:
            md = pq.ParquetFile(fh).metadata
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            rows += rg.num_rows
            for c in range(rg.num_columns):
                if rg.column(c).path_in_schema == "audio.bytes":
                    by += rg.column(c).total_uncompressed_size
    print(pref, rows, "est_hours", round((by - 44 * rows) / 64000 / 3600, 1))
