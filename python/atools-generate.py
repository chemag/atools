#!/usr/bin/env python3

"""generate.py module description.

Runs a generic audio generator.
"""


import argparse
import math
import numpy as np
import os
import sys
import scipy.io.wavfile
import scipy.signal


FILTER_CHOICES = (
    "delta",
    "noise",
    "samples",
)
#    "square",

NOISE_CHOICES = (
    "white",
    "blue",
    "violet",
    "brownian",
    "pink",
)


default_values = {
    "debug": 0,
    "dry_run": False,
    "filter": "delta",
    "samplerate": 16000,
    "duration_sec": None,
    "duration_samples": None,
    "delta_distance_samples": 20,
    "noise_type": "white",
    "max_level": np.iinfo(np.int16).max // 2,
    "samples": [],
    "outfile": None,
}


# generic noise generator
# https://stackoverflow.com/a/67127726
def gen_noise(duration_samples, noise_type, max_level):
    # create white noise signal
    outaud = np.random.randint(
        -max_level, max_level, size=duration_samples, dtype=np.int16
    )
    if noise_type == "white":
        return outaud
    # get the conversion function
    if noise_type == "blue":
        fun = lambda f: np.sqrt(f)
    elif noise_type == "violet":
        fun = lambda f: f
    elif noise_type == "brownian":
        fun = lambda f: 1 / np.where(f == 0, float("inf"), f)
    elif noise_type == "pink":
        fun = lambda f: 1 / np.where(f == 0, float("inf"), np.sqrt(f))
    # create and convert white noise
    X_white = np.fft.rfft(np.random.randn(duration_samples))
    S = fun(np.fft.rfftfreq(duration_samples))
    # normalize S
    S = S / np.sqrt(np.mean(S**2))
    X_shaped = X_white * S
    return np.fft.irfft(X_shaped)


def gen_delta(duration_samples, delta_distance_samples):
    outaud = np.zeros((duration_samples,), dtype=np.int16)
    for i in list(range(0, len(outaud), delta_distance_samples)):
        outaud[i] = np.iinfo(outaud.dtype).max
        # invert one in X samples
        X = 5
        if i % (X * delta_distance_samples) == 0:
            outaud[i] = np.iinfo(outaud.dtype).min
    return outaud


def gen_samples(samples):
    outaud = np.array(samples, dtype=np.int16)
    return outaud


def run_audio_generate(options):
    samplerate = options.samplerate
    duration_samples = (
        len(options.samples)
        if options.filter == "samples"
        else (
            options.duration_samples
            if options.duration_samples is not None
            else (
                samplerate * options.duration_sec
                if options.duration_sec is not None
                else None
            )
        )
    )
    assert (
        duration_samples is not None
    ), f"error: need either --duration-samples or --duration-sec"

    # process the input
    if options.filter == "delta":
        outaud = gen_delta(duration_samples, options.delta_distance_samples)
    elif options.filter == "noise":
        outaud = gen_noise(duration_samples, options.noise_type, options.max_level)
    elif options.filter == "samples":
        outaud = gen_samples(options.samples)
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
        "--samplerate",
        type=int,
        dest="samplerate",
        default=default_values["samplerate"],
        help="Sample Rate",
    )
    parser.add_argument(
        "--duration-samples",
        type=int,
        dest="duration_samples",
        default=default_values["duration_samples"],
        help="Duration (Samples)",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        dest="duration_sec",
        default=default_values["duration_sec"],
        help="Duration (Seconds)",
    )
    parser.add_argument(
        "--delta-distance-samples",
        type=int,
        dest="delta_distance_samples",
        default=default_values["delta_distance_samples"],
        help="Delta Distance (Samples)",
    )
    parser.add_argument(
        "--noise-type",
        action="store",
        type=str,
        dest="noise_type",
        default=default_values["noise_type"],
        choices=NOISE_CHOICES,
        metavar="[%s]"
        % (
            " | ".join(
                NOISE_CHOICES,
            )
        ),
        help="noise type arg",
    )
    parser.add_argument(
        "--max-level",
        type=int,
        dest="max_level",
        default=default_values["max_level"],
        help="Maximum Level of Generated Signal",
    )
    parser.add_argument("--samples", nargs="+", type=int, help="Sample values")
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
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)

    run_audio_generate(options)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
