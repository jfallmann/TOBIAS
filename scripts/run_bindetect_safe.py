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
import faulthandler
import os
import signal
import sys
import traceback
from collections import OrderedDict

# Ensure stdout/stderr are unbuffered/line-buffered so that any output
# (including tracebacks) is flushed even if the process is terminated
# abruptly, instead of being lost in an internal buffer.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Dump a Python traceback to stderr if the process receives a terminating
# signal (e.g. SIGTERM from Slurm on timeout, SIGSEGV) so the cause is
# visible in the log instead of leaving it empty.
faulthandler.enable()
for _sig in ("SIGTERM", "SIGUSR1", "SIGUSR2"):
    if hasattr(signal, _sig):
        try:
            faulthandler.register(getattr(signal, _sig), all_threads=True, chain=True)
        except Exception:
            pass

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


def _preflight_fail(msg):
    print(f"[run_bindetect_safe] PRE-FLIGHT CHECK FAILED: {msg}", flush=True)
    sys.exit(1)


def _preflight_checks(args):
    """Replicate BINDetect's own input validation up front and report any
    failure with a clear, flushed message.

    BINDetect logs errors through a multiprocessing queue/listener and then
    calls sys.exit(1) immediately; if the process tears down before the
    queued log record is drained, the actual reason is lost (only the
    sys.exit(1) is visible). Running the same checks here, printed directly
    and flushed, avoids that race.
    """

    print("[run_bindetect_safe] Running pre-flight checks...", flush=True)

    if len(args.cond_names or []) != len(args.signals):
        _preflight_fail(
            f"--cond_names has {len(args.cond_names or [])} entries but "
            f"--signals has {len(args.signals)} entries; they must match."
        )
    if args.cond_names and len(args.cond_names) != len(set(args.cond_names)):
        _preflight_fail(f"--cond_names contains duplicate values: {args.cond_names}")

    peak_columns = None
    peak_chroms = set()
    peak_regions = []  # (lineno, chrom, start, end)
    n_peaks = 0
    with open(args.peaks) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            n_peaks += 1
            if peak_columns is None:
                peak_columns = len(cols)
            elif len(cols) != peak_columns:
                _preflight_fail(
                    f"--peaks '{args.peaks}' has inconsistent column counts: "
                    f"line 1 has {peak_columns} columns, line {lineno} has "
                    f"{len(cols)} columns."
                )
            peak_chroms.add(cols[0])
            try:
                peak_regions.append((lineno, cols[0], int(cols[1]), int(cols[2])))
            except (ValueError, IndexError):
                _preflight_fail(
                    f"--peaks '{args.peaks}' line {lineno} has non-numeric "
                    f"start/end columns: {cols[:3]}"
                )
    if n_peaks == 0:
        _preflight_fail(f"--peaks file '{args.peaks}' is empty.")
    print(
        f"[run_bindetect_safe]   peaks: {n_peaks} regions, {peak_columns} "
        f"columns, {len(peak_chroms)} unique chromosomes",
        flush=True,
    )

    if args.peak_header:
        with open(args.peak_header) as fh:
            header_list = fh.read().split()
        if len(header_list) != peak_columns:
            _preflight_fail(
                f"--peak_header '{args.peak_header}' has {len(header_list)} "
                f"columns but --peaks has {peak_columns} columns."
            )

    def _check_region_bounds(source_label, chrom_lengths):
        """Mirror TOBIAS' OneRegion.check_boundary(..., action='exit'):
        chrom must exist, start must be >= 0, end must be <= chrom length."""

        missing_chroms = peak_chroms - set(chrom_lengths.keys())
        if missing_chroms:
            _preflight_fail(
                f"{len(missing_chroms)} chromosome name(s) in --peaks are not "
                f"present in {source_label}. Examples missing: "
                f"{sorted(missing_chroms)[:10]}. {source_label} contains e.g.: "
                f"{sorted(chrom_lengths.keys())[:10]}. This usually indicates a "
                "'chr' prefix mismatch (e.g. '1' vs 'chr1') or wrong genome build."
            )
        n_out_of_bounds = 0
        first_offender = None
        for lineno, chrom, start, end in peak_regions:
            length = chrom_lengths[chrom]
            if start < 0 or end > length:
                n_out_of_bounds += 1
                if first_offender is None:
                    first_offender = (lineno, chrom, start, end, length)
        if n_out_of_bounds:
            lineno, chrom, start, end, length = first_offender
            _preflight_fail(
                f"{n_out_of_bounds} region(s) in --peaks fall outside the "
                f"chromosome boundaries defined by {source_label}. First "
                f"offender at line {lineno}: '{chrom}:{start}-{end}' but "
                f"{chrom} is only {length} bp long in {source_label}. This "
                "typically means --peaks was generated against a different "
                "genome build/version than --genome/--signals."
            )

    try:
        import pysam

        fasta = pysam.FastaFile(args.genome)
        fasta_lengths = dict(zip(fasta.references, fasta.lengths))
        fasta.close()
        _check_region_bounds(f"--genome '{args.genome}'", fasta_lengths)
    except ImportError:
        print(
            "[run_bindetect_safe]   WARNING: pysam not available, skipping "
            "genome boundary check",
            flush=True,
        )

    try:
        import pyBigWig

        for sig in args.signals:
            bw = pyBigWig.open(sig)
            bw_lengths = dict(bw.chroms())
            bw.close()
            _check_region_bounds(f"--signals file '{sig}'", bw_lengths)
    except ImportError:
        print(
            "[run_bindetect_safe]   WARNING: pyBigWig not available, skipping "
            "signal boundary check",
            flush=True,
        )

    try:
        from tobias.utils.motifs import MotifList
        from tobias.utils.utilities import expand_dirs

        motif_files = expand_dirs(args.motifs)
        total_motifs = 0
        for f in motif_files:
            try:
                total_motifs += len(MotifList().from_file(f))
            except Exception as e:
                _preflight_fail(f"Could not parse motif file '{f}': {e}")
        print(
            f"[run_bindetect_safe]   motifs: {total_motifs} motifs parsed "
            f"from {len(motif_files)} file(s)",
            flush=True,
        )
    except ImportError as e:
        print(
            f"[run_bindetect_safe]   WARNING: could not import TOBIAS motif "
            f"utilities ({e}), skipping motif parse check",
            flush=True,
        )

    print("[run_bindetect_safe] Pre-flight checks passed.", flush=True)


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
        "--motif_pvalue",
        "--motif-pvalue",
        type=float,
        default=0.0001,
        help="Motif p-value threshold (default: 0.0001)",
    )
    parser.add_argument(
        "--bound_pvalue",
        "--bound-pvalue",
        type=float,
        default=0.001,
        help="Bound p-value threshold (default: 0.001)",
    )
    parser.add_argument(
        "--cluster_threshold",
        "--cluster-threshold",
        type=float,
        default=0.5,
        help="Clustering threshold (default: 0.5)",
    )
    parser.add_argument(
        "--output_peaks",
        "--output-peaks",
        default=None,
        help="Output peak set (bed file path; default: same as --peaks)",
    )
    parser.add_argument(
        "--naming",
        choices=["id", "name", "name_id", "id_name"],
        default="name_id",
        help="Naming convention for TF output files (default: name_id)",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=100,
        help="Split of multiprocessing jobs (default: 100)",
    )
    parser.add_argument(
        "--prefix",
        default="bindetect",
        help="Prefix for overview output files (default: bindetect)",
    )
    parser.add_argument(
        "--pseudo",
        type=float,
        default=None,
        help="Pseudocount for log2fc calculation (default: estimated from data)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Create additional debug PDF with extra plots",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="Verbosity level for logging (default: 0)",
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        return 0

    normalized_argv = _normalize_cli_aliases(sys.argv[1:])
    args = parser.parse_args(normalized_argv)

    # Upstream TOBIAS defines --motifs with nargs="*" (a list) and later does
    # `args.motifs = expand_dirs(args.motifs)`, which iterates over
    # args.motifs assuming it is a list of paths. Our --motifs is declared
    # without nargs, so args.motifs is a plain string; iterating over a
    # string yields individual characters instead of the file path, causing
    # a confusing "No such file or directory: 'T'" error deep inside
    # BINDetect. Normalize to a list here so downstream code behaves as
    # upstream expects.
    if isinstance(args.motifs, str):
        args.motifs = [args.motifs]

    print("[run_bindetect_safe] Parsed arguments:", flush=True)
    for key, value in sorted(vars(args).items()):
        print(f"[run_bindetect_safe]   {key} = {value!r}", flush=True)
    print(
        f"[run_bindetect_safe] TOBIAS_MAX_OPEN_FILES={os.environ.get('TOBIAS_MAX_OPEN_FILES', '128')} "
        f"TOBIAS_RECURSION_LIMIT={os.environ.get('TOBIAS_RECURSION_LIMIT', '20000')}",
        flush=True,
    )

    # Avoid RecursionError in scipy.cluster.hierarchy.dendrogram for
    # large motif collections.
    _ensure_recursion_limit()

    _preflight_checks(args)

    # Monkeypatch both module references used by BINDetect.
    utilities.file_writer = bounded_file_writer
    bindetect.file_writer = bounded_file_writer

    try:
        bindetect.run_bindetect(args)
    except SystemExit as exc:
        print(
            f"[run_bindetect_safe] run_bindetect() called sys.exit({exc.code!r})",
            flush=True,
        )
        raise
    except BaseException:
        print("[run_bindetect_safe] run_bindetect() raised an exception:", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.stderr.flush()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
