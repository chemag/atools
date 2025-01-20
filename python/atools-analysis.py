#!/usr/bin/env python3

"""analyze.py module description.

Runs a generic audio analyzer.
"""


import argparse
import importlib
import math
import numpy as np
import os
import pandas as pd
import sys
import scipy.io.wavfile
import scipy.signal

atools_version = importlib.import_module("atools-version")


ANALYSIS_CHOICES = ("frequency",)

default_values = {
    "debug": 0,
    "dry_run": False,
    "analysis": "frequency",
    "infile": None,
    "outfile": None,
}


# generic frequency analyzer
# https://stackoverflow.com/a/67127726
def analyze_frequencies(infile):
    # read the input
    samplerate, inaud = scipy.io.wavfile.read(infile)
    # calculate the Discrete Fourier Transform sample frequencies
    freq_list = np.fft.rfftfreq(len(inaud)) * samplerate
    power_values = np.abs(np.fft.rfft(inaud))
    freq_df = pd.DataFrame(
        columns=["frequency", "power"], data=zip(freq_list, power_values)
    )
    return freq_df


def run_audio_analyze(options):
    # process the input
    if options.analysis == "frequency":
        freq_df = analyze_frequencies(options.infile)
        freq_df.to_csv(options.outfile, index=False)


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namespace - An argparse.ArgumentParser-analyzed option object
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
        "--analysis",
        action="store",
        type=str,
        dest="analysis",
        default=default_values["analysis"],
        choices=ANALYSIS_CHOICES,
        metavar="[%s]"
        % (
            " | ".join(
                ANALYSIS_CHOICES,
            )
        ),
        help="analysis arg",
    )
    parser.add_argument(
        "-i",
        "--infile",
        dest="infile",
        type=str,
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-o",
        "--outfile",
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
    # get infile/outfile
    if options.infile is None or options.infile == "-":
        options.infile = "/dev/fd/0"
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)

    run_audio_analyze(options)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
