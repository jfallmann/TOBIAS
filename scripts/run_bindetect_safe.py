#!/usr/bin/env python3

"""
Run TOBIAS BINDetect with a bounded-file-handle writer.

Why this exists:
- Upstream BINDetect's file_writer pre-opens all temp files assigned to a writer process.
- For large motif collections this can exceed OS file descriptor limits (Errno 24).

This wrapper monkeypatches BINDetect's writer function to:
1) truncate/create all target files once,
2) write with a bounded LRU cache of open handles.

The maximum number of concurrently open handles can be controlled by
environment variable TOBIAS_MAX_OPEN_FILES (default: 128).

For very large motif sets, SciPy dendrogram plotting may exceed Python's
default recursion depth. This wrapper raises recursion depth using
TOBIAS_RECURSION_LIMIT (default: 20000).
"""

import argparse
import os
import sys
from collections import OrderedDict

import tobias.tools.bindetect as bindetect
import tobias.utils.utilities as utilities


def _normalize_cli_aliases(argv):
    """Translate legacy underscore argument names to dash-style argparse names."""

    aliases = {
        "--cond_names": "--cond-names",
        "--peak_header": "--peak-header",
        "--time_series": "--time-series",
        "--motif_pvalue": "--motif-pvalue",
        "--bound_pvalue": "--bound-pvalue",
        "--cluster_threshold": "--cluster-threshold",
        "--output_peaks": "--output-peaks",
        "--norm_off": "--norm-off",
        "--skip_excel": "--skip-excel",
    }
    return [aliases.get(token, token) for token in argv]


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _ensure_recursion_limit():
    """Raise recursion limit for large dendrogram plotting workloads."""

    requested = max(
        2000, _safe_int(os.environ.get("TOBIAS_RECURSION_LIMIT", "20000"), 20000)
    )
    current = sys.getrecursionlimit()
    if requested > current:
        sys.setrecursionlimit(requested)


def bounded_file_writer(q, key_file_dict, args):
    """Drop-in replacement for tobias.utils.utilities.file_writer.

    Keeps only a bounded number of file handles open at any time to avoid
    hitting per-process file descriptor limits.
    """

    max_open = max(8, _safe_int(os.environ.get("TOBIAS_MAX_OPEN_FILES", "128"), 128))

    # Ensure files exist and are truncated before appending chunks.
    unique_files = set(key_file_dict.values())
    for fil in unique_files:
        os.makedirs(os.path.dirname(fil), exist_ok=True)
        with open(fil, "w"):
            pass

    open_handles = OrderedDict()  # path -> handle, LRU ordering

    def get_handle(path):
        handle = open_handles.get(path)
        if handle is not None:
            open_handles.move_to_end(path)
            return handle

        if len(open_handles) >= max_open:
            _, oldest = open_handles.popitem(last=False)
            oldest.close()

        handle = open(path, "a")
        open_handles[path] = handle
        return handle

    try:
        while True:
            key, content = q.get()
            if key is None:
                break

            if key not in key_file_dict:
                raise KeyError(f"Received unknown writer key: {key}")

            outfile = key_file_dict[key]
            handle = get_handle(outfile)
            handle.write(content)

    finally:
        for handle in open_handles.values():
            try:
                handle.close()
            except Exception:
                pass

    return 0


def main():
    # Create argument parser locally since add_bindetect_arguments may not exist
    parser = argparse.ArgumentParser(
        description="Run TOBIAS BINDetect with bounded file handle management"
    )

    # Define core arguments based on TOBIAS BINDetect CLI
    parser.add_argument("--motifs", required=True, help="Motif file (FASTA format)")
    parser.add_argument(
        "--signals", nargs="+", required=True, help="Signal files (bigWig format)"
    )
    parser.add_argument("--genome", required=True, help="Genome FASTA file")
    parser.add_argument("--peaks", required=True, help="Peak file (BED format)")
    parser.add_argument(
        "--peak_header", "--peak-header", required=True, help="Peak header file"
    )
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--cores", "--threads", type=int, default=1, help="Number of cores to use"
    )
    parser.add_argument(
        "--cond_names", "--cond-names", nargs="+", help="Condition names"
    )

    # Optional arguments
    parser.add_argument(
        "--norm_off", "--norm-off", action="store_true", help="Turn off normalization"
    )
    parser.add_argument(
        "--skip_excel", "--skip-excel", action="store_true", help="Skip Excel output"
    )
    parser.add_argument(
        "--time_series", "--time-series", action="store_true", help="Time series mode"
    )
    parser.add_argument(
        "--motif_pvalue", "--motif-pvalue", type=float, help="Motif p-value threshold"
    )
    parser.add_argument(
        "--bound_pvalue", "--bound-pvalue", type=float, help="Bound p-value threshold"
    )
    parser.add_argument(
        "--cluster_threshold",
        "--cluster-threshold",
        type=float,
        help="Clustering threshold",
    )
    parser.add_argument(
        "--output_peaks",
        "--output-peaks",
        action="store_true",
        help="Output peak files",
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        return 0

    normalized_argv = _normalize_cli_aliases(sys.argv[1:])
    args = parser.parse_args(normalized_argv)

    # Avoid RecursionError in scipy.cluster.hierarchy.dendrogram for
    # large motif collections.
    _ensure_recursion_limit()

    # Monkeypatch both module references used by BINDetect.
    utilities.file_writer = bounded_file_writer
    bindetect.file_writer = bounded_file_writer

    bindetect.run_bindetect(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
