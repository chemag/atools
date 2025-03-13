#!/usr/bin/env python3

"""atools-enctool.py module description.

This is a tool to test audio encoders.
"""


import argparse
import importlib
import itertools
import pandas as pd
import sys
import tempfile

# atools_analysis = importlib.import_module("atools-analysis")
atools_common = importlib.import_module("atools-common")
atools_version = importlib.import_module("atools-version")

OPUS_COMPLEXITY_LIST = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
OPUS_CONTENT_MODE_LIST = ["music", "speech"]
OPUS_BITRATE_KBPS_LIST = ["8", "16", "32", "64", "128", "256"]
OPUS_FRAME_SIZE_MS_LIST = ["2.5", "5", "10", "20", "40", "60"]

# OPUS_COMPLEXITY_LIST = list(str(i) for i in range(0, 11, 5))
# OPUS_BITRATE_KBPS_LIST = ["8", "32", "128"]
# OPUS_FRAME_SIZE_MS_LIST = ["2.5", "10", "40"]

default_values = {
    "debug": 0,
    "dry_run": False,
    "complexity_list": OPUS_COMPLEXITY_LIST,
    "content_mode_list": OPUS_CONTENT_MODE_LIST,
    "bitrate_kbps_list": OPUS_BITRATE_KBPS_LIST,
    "frame_size_ms_list": OPUS_FRAME_SIZE_MS_LIST,
    "infile_list": None,
    "outfile": None,
    "logfile": None,
}

COLUMN_LIST = [
    "infile",
    "complexity",
    "mode",
    "bitrate_kbps",
    "frame_size_ms",
]


def process_file(
    infile,
    complexity,
    content_mode,
    bitrate_kbps,
    frame_size_ms,
    debug,
):
    # create the command
    outfile = tempfile.NamedTemporaryFile(
        prefix="atools.enctools.", suffix=".opus"
    ).name
    command = f"opusenc --comp {complexity} --{content_mode} --bitrate {bitrate_kbps} --framesize {frame_size_ms} {infile} {outfile}"
    # run the command
    returncode, out, err, stats = atools_common.run(
        command, logfd=None, debug=debug, gnu_time=True
    )
    assert returncode == 0, f"error: {out = } {err = }"
    return stats


def process_data(
    infile_list,
    complexity_list,
    content_mode_list,
    bitrate_kbps_list,
    frame_size_ms_list,
    outfile,
    debug,
):
    df = None
    for (
        infile,
        complexity,
        content_mode,
        bitrate_kbps,
        frame_size_ms,
    ) in itertools.product(
        infile_list,
        complexity_list,
        content_mode_list,
        bitrate_kbps_list,
        frame_size_ms_list,
    ):
        # 1. encode the file
        stats = process_file(
            infile, complexity, content_mode, bitrate_kbps, frame_size_ms, debug
        )
        # 2. convert the stats
        stats_value_list = list(stats.values())
        # 3. measure the quality
        # 4. store the tuple
        if df is None:
            stats_column_list = list(f"stats:{key}" for key in stats.keys())
            df = pd.DataFrame(columns=COLUMN_LIST + stats_column_list)
        df.loc[df.size] = (
            infile,
            complexity,
            content_mode,
            bitrate_kbps,
            frame_size_ms,
            *stats_value_list,
        )

    # 3. write the results
    df.to_csv(outfile, index=False)


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namespace - An argparse.ArgumentParser-generated option object
    """
    # init parser
    # usage = 'usage: %prog [options] arg1 arg2'
    # parser = argparse.OptionParser(usage=usage)
    # parser.print_help() to get argparse.usage (large help)
    # parser.print_usage() to get argparse.usage (just usage line)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=atools_version.__version__,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="count",
        dest="debug",
        default=default_values["debug"],
        help="Increase verbosity (use multiple times for more)",
    )
    parser.add_argument(
        "--quiet",
        action="store_const",
        dest="debug",
        const=-1,
        help="Zero verbosity",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        default=default_values["dry_run"],
        help="Dry run",
    )
    parser.add_argument(
        "--complexity-list",
        action="store",
        type=str,
        dest="complexity_list",
        default=default_values["complexity_list"],
        help="Complexity list (comma-separated list)",
    )
    parser.add_argument(
        "--content-mode-list",
        action="store",
        type=str,
        dest="content_mode_list",
        default=default_values["content_mode_list"],
        help="Content mode list (comma-separated list)",
    )
    parser.add_argument(
        "--bitrate-kbps-list",
        action="store",
        type=str,
        dest="bitrate_kbps_list",
        default=default_values["bitrate_kbps_list"],
        help="Bitrate (kbps) list (comma-separated list)",
    )
    parser.add_argument(
        "--frame-size-ms-list",
        action="store",
        type=str,
        dest="frame_size_ms_list",
        default=default_values["frame_size_ms_list"],
        help="Frame Size (ms) list (comma-separated list)",
    )

    parser.add_argument(
        dest="infile_list",
        type=str,
        nargs="+",
        default=default_values["infile_list"],
        metavar="input-file-list",
        help="input file list",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        action="store",
        dest="outfile",
        type=str,
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    # get outfile
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(f"debug: {options}")
    # create configuration
    # process infile
    process_data(
        options.infile_list,
        options.complexity_list,
        options.content_mode_list,
        options.bitrate_kbps_list,
        options.frame_size_ms_list,
        options.outfile,
        options.debug,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
