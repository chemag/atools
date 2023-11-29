#!/usr/bin/env python
# (c) Facebook, Inc. and its affiliates.


import argparse
import numpy as np
import scipy.io.wavfile
import scipy.signal
import sys

SIGNAL_CHOICES = ("multitone",)

default_values = {
    "debug": 0,
    "samplerate": 16000,
    "active_ratio": 0.1,
    "duration_sec": 1,
    "f0": 50,
    "f1": 12800,
    "scale": 0.7,
    "signal": "multitone",
    "outfile": None,
}


# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.chirp.html
# https://scipy-cookbook.readthedocs.io/items/FrequencySweptDemo.html
def generate_chirp(duration_sec, f0, f1, samplerate, scale):
    t = np.linspace(0, duration_sec, duration_sec * samplerate)
    w = scipy.signal.chirp(t, f0=f0, t1=duration_sec, f1=f1, method="linear")
    outarray = np.zeros(len(w), dtype=np.int16)
    for i in range(w.shape[0]):
        outarray[i] = int(w[i] * (2**15) * scale)
    return outarray


# generate an X-second multitone from f0 Hz to f1 Hz
def generate_multitone(samplerate, duration_sec, active_ratio, f0, f1, scale, outfile):
    # generate the active signal in different octaves
    freq = f0
    signal_array = []
    active_duration_sec = 1.0 * active_ratio
    while freq <= f1:
        signal_array.append(generate_sin(active_duration_sec, freq, samplerate, scale))
        freq *= 2

    # average the signals in the array
    active_signal = np.zeros(signal_array[0].shape, dtype=np.int32)
    for signal in signal_array:
        active_signal += signal
    active_signal //= len(signal_array)
    active_signal = active_signal.astype("int16")

    # generate the non-active signal
    silence_duration_sec = 1.0 * (1 - active_ratio)
    silence_signal = generate_silence(silence_duration_sec, samplerate)

    # concatenate active and non-active signals
    one_second_signal = np.concatenate((active_signal, silence_signal))

    # repeat the signal
    final_signal = one_second_signal
    for _ in range(int(duration_sec) - 1):
        final_signal = np.concatenate((final_signal, one_second_signal))

    # write to wav file
    scipy.io.wavfile.write(outfile, samplerate, final_signal)


# https://docs.scipy.org/doc/numpy/reference/generated/numpy.sin.html
def generate_sin(duration_sec, f, samplerate, scale):
    t = np.arange(samplerate * duration_sec)
    samples = (2**15) * scale * np.sin(2 * np.pi * t * f / samplerate)
    return samples.astype(np.int16)


def generate_silence(duration_sec, samplerate):
    shape = (int(duration_sec * samplerate),)
    silence = np.zeros(shape, dtype=np.int16)
    return silence


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
        "--samplerate",
        action="store",
        type=int,
        dest="samplerate",
        default=default_values["samplerate"],
        metavar="SAMPLERATE",
        help="use SAMPLERATE samplerate",
    )
    parser.add_argument(
        "--duration-sec",
        action="store",
        type=float,
        dest="duration_sec",
        default=default_values["duration_sec"],
        metavar="DURATION_SEC",
        help="use DURATION_SEC duration_sec",
    )
    parser.add_argument(
        "--active-ratio",
        action="store",
        type=float,
        dest="active_ratio",
        default=default_values["active_ratio"],
        metavar="ACTIVE-RATIO",
        help="use active_ratio",
    )
    parser.add_argument(
        "--f0",
        action="store",
        type=int,
        dest="f0",
        default=default_values["f0"],
        metavar="F0",
        help="use F0 for start chirp frequency",
    )
    parser.add_argument(
        "--f1",
        action="store",
        type=int,
        dest="f1",
        default=default_values["f1"],
        metavar="F1",
        help="use F1 for end chirp frequency",
    )
    parser.add_argument(
        "--scale",
        action="store",
        type=float,
        dest="scale",
        default=default_values["scale"],
        metavar="SCALE",
        help="use scale",
    )
    parser.add_argument(
        "--signal",
        action="store",
        type=str,
        dest="signal",
        default=default_values["signal"],
        choices=SIGNAL_CHOICES,
        metavar="[%s]"
        % (
            " | ".join(
                SIGNAL_CHOICES,
            )
        ),
        help="signal arg",
    )
    parser.add_argument(
        "outfile",
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
    if options.outfile == "-":
        options.outfile = sys.stdout
    # print results
    if options.debug > 0:
        print(options)
    # do something
    if options.signal == "multitone":
        generate_multitone(
            options.samplerate,
            options.duration_sec,
            options.active_ratio,
            options.f0,
            options.f1,
            options.scale,
            options.outfile,
        )
    else:
        print("error: invalid signal: %s" % options.signal)
        sys.exit(-1)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
