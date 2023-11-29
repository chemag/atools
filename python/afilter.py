#!/usr/bin/env python3

"""filter.py module description.

Runs a generic audio filter.
"""


import argparse
import math
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
    "add",
    "diff",
)

default_values = {
    "debug": 0,
    "dry_run": False,
    "filter": "copy",
    "infile": None,
    "infile2": None,
    "outfile": None,
}


def invert_phase(inaud, **kwargs):
    outaud = -inaud
    return outaud


def get_rms(inaud):
    # RMS = \sqrt((\sum_i sample[i] ** 2) / n)
    square_sum = 0
    inlen = len(inaud)
    max_value = 0
    for sample in inaud:
        square_sum += sample**2
    rms = math.sqrt(square_sum / inlen)
    # dBFS = 20 * math.log10(RMS / max_sample_value)
    max_sample_value = max(abs(np.amax(inaud)), abs(np.amin(inaud)))
    if max_sample_value != 0:
        dbfs = 20 * math.log10(rms / max_sample_value)
    else:
        dbfs = math.nan
    return rms, dbfs


def print_stats(inaud, samplerate):
    print(f"samplerate: {samplerate}")
    print(f"num_samples: {len(inaud)}")
    # add some statistics on the audio signal
    mean = np.mean(inaud)
    stddev = np.std(inaud)
    print(f"mean: {mean}")
    print(f"stddev: {stddev}")
    max_value = np.amax(inaud)
    min_value = np.amin(inaud)
    print(f"max_value: {max_value}")
    print(f"min_value: {min_value}")
    rms, dbfs = get_rms(inaud)
    print(f"RMS: {rms}")
    print(f"dBFS: {dbfs}")


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


# adds inputs with saturation (not mixing)
def add_inputs(in1aud, in2aud, **kwargs):
    # start with int32 to allow additions
    in1len = len(in1aud)
    in2len = len(in2aud)
    outlen = max(in1len, in2len)
    outaud = np.zeros((outlen,), dtype=np.int32)
    # add the inputs
    for i in range(outlen):
        outaud[i] += in1aud[i] if i < in1len else 0
        outaud[i] += in2aud[i] if i < in2len else 0
    # saturate the output
    outaud[outaud > np.iinfo(np.int16).max] = np.iinfo(np.int16).max
    outaud[outaud < np.iinfo(np.int16).min] = np.iinfo(np.int16).min
    return outaud.astype(np.int16)


# diffs inputs with saturation (not mixing)
def diff_inputs(in1aud, in2aud, **kwargs):
    # start with int32 to allow additions
    in1len = len(in1aud)
    in2len = len(in2aud)
    outlen = max(in1len, in2len)
    outaud = np.zeros((outlen,), dtype=np.int32)
    # diff the inputs
    for i in range(outlen):
        outaud[i] += in1aud[i] if i < in1len else 0
        outaud[i] -= in2aud[i] if i < in2len else 0
    # saturate the output
    outaud[outaud > np.iinfo(np.int16).max] = np.iinfo(np.int16).max
    outaud[outaud < np.iinfo(np.int16).min] = np.iinfo(np.int16).min
    return outaud.astype(np.int16)


def run_audio_filter(options):
    # open the input
    samplerate, inaud = scipy.io.wavfile.read(options.infile)
    if options.infile2 is not None:
        samplerate2, in2aud = scipy.io.wavfile.read(options.infile2)
        # TODO(chema): fix this?
        assert (
            samplerate == samplerate2
        ), f"error: both input files must have the same sample rate ({samplerate} != {samplerate2})"
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
    elif options.filter == "add":
        outaud = add_inputs(inaud, in2aud)
    elif options.filter == "diff":
        outaud = diff_inputs(inaud, in2aud)
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
        "-j",
        "--infile2",
        dest="infile2",
        type=str,
        default=default_values["infile2"],
        metavar="input-file2",
        help="input file2",
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
