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
    "cut",
    "stats",
    "noise",
    "phase-invert",
    "shift",
    "add",
    "append",
    "diff",
)

default_values = {
    "debug": 0,
    "dry_run": False,
    "filter": "copy",
    "shift": 0,
    "start_sample": 0,
    "end_sample": -1,
    "duration_samples": -1,
    "infile": None,
    "infile2": None,
    "outfile": None,
}


def cut_signal(inaud, start_sample, end_sample, duration_samples):
    # use the input length as default end sample
    end_sample = len(inaud) if end_sample == -1 else end_sample
    # get duration in samples
    if duration_samples == -1:
        # get duration from start and end sample
        duration_samples = end_sample - start_sample + 1
    else:
        end_sample = start_sample + duration_samples - 1
    # cut the input
    outaud = np.zeros((duration_samples,), dtype=np.int16)
    outaud = inaud[start_sample : end_sample + 1]
    return outaud


def invert_phase(inaud, **kwargs):
    outaud = -inaud
    return outaud


def shift_signal(inaud, shift):
    # start with the same length and type
    outaud = np.zeros(inaud.shape, dtype=inaud.dtype)
    # process shifts by sign
    if shift > 0:
        outaud[shift:] = inaud[:-shift]
    elif shift < 0:
        outaud[:shift] = inaud[-shift:]
    return outaud


def get_rms(inaud):
    # add up all the input samples
    total_linear = 0
    total_abs = 0
    total_square = 0
    for i in range(len(inaud)):
        total_linear += inaud[i]
        total_abs += int(inaud[i]) if inaud[i] > 0 else -int(inaud[i])
        total_square += 1 * inaud[i] * inaud[i]
    total_linear /= len(inaud)
    total_abs /= len(inaud)
    total_square /= len(inaud)
    total_square = math.sqrt(total_square)
    # normalize values
    normalized_linear = total_linear / np.iinfo(inaud.dtype).max
    normalized_abs = total_abs / np.iinfo(inaud.dtype).max
    normalized_square = total_square / np.iinfo(inaud.dtype).max
    if normalized_square > 0:
        rms = 20 * math.log10(normalized_square / 1.0)
    else:
        rms = math.nan
    return normalized_linear, normalized_abs, normalized_square, rms


def print_stats(inaud, samplerate):
    print(f"samplerate: {samplerate}")
    print(f"num_samples: {len(inaud)}")
    # add some statistics on the audio signal
    max_value = np.amax(inaud)
    min_value = np.amin(inaud)
    print(f"max_value: {max_value}")
    print(f"min_value: {min_value}")
    mean_abs_value = np.mean(np.abs(inaud))
    stddev = np.std(inaud)
    print(f"mean_abs_value: {mean_abs_value}")
    print(f"stddev: {stddev}")
    normalized_linear, normalized_abs, normalized_square, rms = get_rms(inaud)
    print(f"normalized_linear: {normalized_linear}")
    print(f"normalized_abs: {normalized_abs}")
    print(f"normalized_square: {normalized_square}")
    print(f"rms: {rms}")


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


def append_inputs(in1aud, in2aud, **kwargs):
    # start with int32 to allow additions
    in1len = len(in1aud)
    in2len = len(in2aud)
    outlen = in1len + in2len
    outaud = np.zeros((outlen,), dtype=np.int16)
    # append the inputs
    outaud[0:in1len] = in1aud
    outaud[in1len:] = in2aud
    return outaud


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


def get_channel(inaud, index):
    if inaud is None:
        return inaud
    nchannels = inaud.shape[1] if len(inaud.shape) > 1 else 1
    if index == 0 and nchannels == 1:
        return inaud
    return inaud[:, index]


def process_input(in1aud, in2aud, samplerate, options):
    out_channels = []
    # calculate the number of output channels
    num_in_channels1 = in1aud.shape[1] if len(in1aud.shape) > 1 else 1
    num_in_channels2 = (
        0 if in2aud is None else (in2aud.shape[1] if len(in2aud.shape) > 1 else 1)
    )
    num_out_channels = max(num_in_channels1, num_in_channels2)
    for index in range(num_in_channels1):
        in_channel1 = get_channel(in1aud, index)
        in_channel2 = get_channel(in2aud, index)
        out_channels.append(
            process_input_channel(in_channel1, in_channel2, samplerate, options)
        )
    # put all the output channels together
    if len(out_channels) == 1:
        return out_channels[0]
    return np.stack((out_channels), axis=1)


def process_input_channel(inaud, in2aud, samplerate, options):
    if options.filter == "copy":
        outaud = inaud
    elif options.filter == "cut":
        outaud = cut_signal(
            inaud, options.start_sample, options.end_sample, options.duration_samples
        )
    elif options.filter == "stats":
        print_stats(inaud, samplerate)
        return None
    elif options.filter == "noise":
        outaud = add_noise(inaud)
    elif options.filter == "phase-invert":
        outaud = invert_phase(inaud)
    elif options.filter == "shift":
        outaud = shift_signal(inaud, options.shift)
    elif options.filter == "add":
        outaud = add_inputs(inaud, in2aud)
    elif options.filter == "append":
        outaud = append_inputs(inaud, in2aud)
    elif options.filter == "diff":
        outaud = diff_inputs(inaud, in2aud)
    return outaud


def run_audio_filter(options):
    # open the input
    samplerate, inaud = scipy.io.wavfile.read(options.infile)
    in2aud = None
    if options.infile2 is not None:
        samplerate2, in2aud = scipy.io.wavfile.read(options.infile2)
        # TODO(chema): fix this?
        assert (
            samplerate == samplerate2
        ), f"error: both input files must have the same sample rate ({samplerate} != {samplerate2})"
    # process the input
    outaud = process_input(inaud, in2aud, samplerate, options)
    # write the output
    if outaud is not None:
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
        "--shift",
        type=int,
        dest="shift",
        default=default_values["shift"],
        help="Shift Amount",
    )
    parser.add_argument(
        "--start-sample",
        type=int,
        dest="start_sample",
        default=default_values["start_sample"],
        help="Cut start (in samples)",
    )
    parser.add_argument(
        "--end-sample",
        type=int,
        dest="end_sample",
        default=default_values["end_sample"],
        help="Cut end (in samples)",
    )
    parser.add_argument(
        "--duration-samples",
        type=int,
        dest="duration_samples",
        default=default_values["duration_samples"],
        help="Cut duration (in samples)",
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
