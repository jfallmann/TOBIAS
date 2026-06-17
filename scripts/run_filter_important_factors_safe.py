#!/usr/bin/env python3

"""
Wrapper around filter_important_factors.py that handles NA values in
bindetect_results.txt.

Why this exists:
- BINDetect writes 'NA' for p-values when comparisons cannot be computed.
- filter_important_factors.py calls float() on every value without guarding
  against NA, causing a ValueError crash.
- This wrapper replaces NA with 0 in numeric columns before running the filter,
  then passes through to filter_important_factors.py.
"""

import argparse
import os
import subprocess
import sys
import tempfile


def replace_na_in_numeric_columns(src_path, dst_path):
    """
    Copy bindetect results to dst_path, replacing 'NA' with '0' in any
    column that is otherwise numeric.  Non-numeric columns (name, cluster,
    output_prefix …) are left untouched.
    """
    with open(src_path, "r") as fh:
        lines = fh.readlines()

    if not lines:
        with open(dst_path, "w") as fh:
            pass
        return

    header = lines[0].rstrip("\n").split("\t")
    rows = [line.rstrip("\n").split("\t") for line in lines[1:]]

    # Identify which columns are numeric (ignore NA for this check)
    numeric_cols = set()
    for col_idx in range(len(header)):
        values = []
        for row in rows:
            if col_idx < len(row) and row[col_idx] not in ("NA", ""):
                values.append(row[col_idx])
        if values:
            try:
                [float(v) for v in values]
                numeric_cols.add(col_idx)
            except ValueError:
                pass

    # Rewrite, replacing NA with 0 only in numeric columns
    with open(dst_path, "w") as fh:
        fh.write(lines[0])
        for row in rows:
            new_row = []
            for col_idx, val in enumerate(row):
                if col_idx in numeric_cols and val in ("NA", ""):
                    new_row.append("0")
                else:
                    new_row.append(val)
            fh.write("\t".join(new_row) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run filter_important_factors.py safely with NA handling"
    )
    parser.add_argument(
        "-in", dest="input", required=True, help="Input bindetect_results.txt"
    )
    parser.add_argument(
        "-filter",
        dest="filter",
        required=True,
        help="Filter value passed to filter_important_factors.py",
    )
    parser.add_argument("-o", dest="output", required=True, help="Output file")

    args = parser.parse_args()

    # Write a cleaned version of the input to a temp file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="bindetect_nona_")
    os.close(tmp_fd)
    try:
        replace_na_in_numeric_columns(args.input, tmp_path)

        cmd = [
            "filter_important_factors.py",
            "-in",
            tmp_path,
            "-filter",
            str(args.filter),
            "-o",
            args.output,
        ]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
