#!/usr/bin/env python3

"""filter.py module description.

Runs a generic audio filter.
"""


import argparse
import numpy as np
import os
import sys
import scipy.io.wavfile
import scipy.signal


FILTER_CHOICES = (
    "copy",
    "stats",
    "noise",
    "phase-invert",
)

default_values = {
    "debug": 0,
    "dry_run": False,
    "filter": "copy",
    "infile": None,
    "outfile": None,
}


def invert_phase(inaud, **kwargs):
    outaud = -inaud
    return outaud


def print_stats(inaud, samplerate):
    print(f"samplerate: {samplerate}")
    print(f"num_samples: {len(inaud)}")


def add_noise(inaud, **kwargs):
    # get optional input parameters
    random_max = np.iinfo(inaud.dtype).max // 2
    random_max = kwargs.get("random_max", random_max)
    # create noise signal
    # note that we need to upgrade the signal to int32
    noiseaud = np.random.randint(
        -random_max, random_max, size=inaud.shape, dtype=np.int32
    )
    outaud = inaud + noiseaud
    outaud[outaud > np.iinfo(np.int16).max] = np.iinfo(np.int16).max
    outaud[outaud < np.iinfo(np.int16).min] = np.iinfo(np.int16).min
    return outaud.astype(np.int16)


def run_audio_filter(options):
    # open the input
    samplerate, inaud = scipy.io.wavfile.read(options.infile)
    # process the input
    if options.filter == "copy":
        outaud = inaud
    elif options.filter == "stats":
        print_stats(inaud, samplerate)
        return
    elif options.filter == "noise":
        outaud = add_noise(inaud)
    elif options.filter == "phase-invert":
        outaud = invert_phase(inaud)
    # write the output
    scipy.io.wavfile.write(options.outfile, samplerate, outaud)


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
        action="store_true",
        dest="version",
        default=False,
        help="Print version",
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
        "--filter",
        action="store",
        type=str,
        dest="filter",
        default=default_values["filter"],
        choices=FILTER_CHOICES,
        metavar="[%s]"
        % (
            " | ".join(
                FILTER_CHOICES,
            )
        ),
        help="filter arg",
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
    if options.version:
        return options
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

    run_audio_filter(options)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
