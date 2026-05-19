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
"""

import argparse
import os
import sys
from collections import OrderedDict

import tobias.tools.bindetect as bindetect
import tobias.utils.utilities as utilities
from tobias.parsers import add_bindetect_arguments


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


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
    parser = argparse.ArgumentParser()
    parser = add_bindetect_arguments(parser)

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        return 0

    args = parser.parse_args()

    # Monkeypatch both module references used by BINDetect.
    utilities.file_writer = bounded_file_writer
    bindetect.file_writer = bounded_file_writer

    bindetect.run_bindetect(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
