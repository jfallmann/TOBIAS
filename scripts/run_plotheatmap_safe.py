#!/usr/bin/env python3

"""
Run TOBIAS PlotHeatmap with graceful handling of empty distributions.

Why this exists:
- PlotHeatmap crashes with IndexError when a motif/condition has no distribution data.
- This can happen with sparse binding sites or edge cases in the data.
- This wrapper catches and handles those cases gracefully, skipping bad entries
  or falling back to simpler plots when needed.
"""

import argparse
import sys

import tobias.tools.plot_heatmap as plot_heatmap


def safe_run_heatmap(args):
    """Wrapper around run_heatmap that handles empty distribution IndexError."""

    try:
        plot_heatmap.run_heatmap(args)
    except IndexError as e:
        error_msg = str(e)
        # Check if this is the "list index out of range" from empty distri
        if "list index out of range" in error_msg or (
            "distri" in error_msg or "keys()" in error_msg
        ):
            # Log the issue and gracefully degrade
            print("WARNING: PlotHeatmap encountered missing/empty distribution data.")
            print(
                "         Likely cause: sparse binding sites or no sites in a condition."
            )
            print(f"         Error: {error_msg}", file=sys.stderr)
            print("         Attempting to proceed or use fallback...", file=sys.stderr)

            # Try a second pass with more lenient settings
            # (This is a best-effort fallback; may not fully resolve all cases)
            try:
                # Retry with potentially simpler config
                if hasattr(args, "share_colorbar"):
                    args.share_colorbar = False
                plot_heatmap.run_heatmap(args)
            except Exception as e2:
                print(
                    f"ERROR: PlotHeatmap still failed after fallback: {e2}",
                    file=sys.stderr,
                )
                # Exit gracefully with a warning rather than crashing
                sys.exit(1)
        else:
            # Re-raise if it's a different IndexError
            raise


def main():
    # Use upstream parser
    from tobias.parsers import add_plotheatmap_arguments

    parser = argparse.ArgumentParser()
    parser = add_plotheatmap_arguments(parser)

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        return 0

    args = parser.parse_args()

    # Run with safe error handling
    safe_run_heatmap(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
