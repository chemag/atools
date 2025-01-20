#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates.

import argparse
import importlib
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf

atools_common = importlib.import_module("atools-common")
atools_playrec = importlib.import_module("atools-playrec")

gain = -1


def findDelayInFile(noisy_path, ref_path, threshold, verbose):
    """Find first two signals in an audio file

    Calculates the difference between the first two signals
    matching reference to sufficient degree.
    intpuArgst:
        noisy_path: path to wavfile to be searched
        ref_path: path to wavfiile containing the signal
        threshold: 0-100, correlations above this are considered
        verbose: if True the correlations values will be printed out
                 and a grap[h will be drawn showing the correlation
                 and wave forms

    Returns:
        delay: in samplse,
        delay: in ms,
        correlation: for first signal,
        correlation: for last signal
    """

    noisy = sf.SoundFile(noisy_path, "r")
    ref = sf.SoundFile(ref_path, "r")

    noisy_data = noisy.read()
    ref_data = ref.read()
    return findDelayInData(noisy_data, ref_data, threshold, noisy.samplerate, verbose)


def findDelayInData(noisy_data, ref_data, threshold, samplerate, verbose):
    """Find first two signals in audio data

    Calculates the difference between the first two signals
    matching reference to sufficient degree.
    Args:
        noisy_data: numpy data to be searched
        ref_data: numpy data containing the signal
        threshold: 0-100, correlations above this are considered
        samplerate: needed to calculate time
        verbose: if True the correlations values will be printed out
                 and a grap[h will be drawn showing the correlation
                 and wave forms

    Returns:
        delay: in samplse,
        delay: in ms,
        correlation: for first signal,
        correlation: for last signal
    """

    global gain
    # Check audio level once, if to low warn but try to gain up.
    if gain == -1:
        max_level = abs(np.max(noisy_data))
        gain = 0.7 / max_level  #  -3dB

    noisy_data = noisy_data * gain
    data = atools_common.find_markers(ref_data, noisy_data, threshold, samplerate, verbose)
    if len(data) > 1:
        diff = data.loc[1, "sample"] - data.loc[0, "sample"]
        return [
            diff,
            round(diff / samplerate * 1000),
            data.loc[0, "correlation"],
            data.loc[1, "correlation"],
        ]

        return None


def measureDelay(impulse, threshold, delay, seconds, outputcsv, save, verbose=False):
    """Record and calculate delay times from echoes

    Starts a recortding and play signal with an even delay.
    The audio is used to find echoes and the delay is the time
    between the first and the second signal in the recording.

    Args:
        impulse: path to the signal being used
        threshold: 0-100, correlations above this are considered
        delay: time between signals in seconds
        outputcsv: write the results to this csv
        save: save the recordded data to a cocatenated file
        verbose: if True the correlations values will be printed out
                 and a grap[h will be drawn showing the correlation
                 and wave forms
    """

    ref = None
    if impulse[-4:] == ".wav":
        ref = sf.SoundFile(impulse, "r")
        ref_data = ref.read()
    else:
        print("Only wav file supported")
        exit(0)

    outputfile = None
    failed_counter = 0
    delays = []
    start = time.time()
    now = start
    tmpfile = outputcsv + "_.wav"
    while now - start < seconds:
        atools_playrec.play_and_record(impulse, tmpfile, delay)
        noisy = sf.SoundFile(tmpfile, "r")
        noisy_data = noisy.read()
        data = findDelayInData(noisy_data, ref_data, threshold, ref.samplerate, verbose)
        if data is not None:
            delays.append(data)
        else:
            failed_counter += 1

        if (len(save) > 0) & (outputfile is None):
            outputfile = sf.SoundFile(
                save, "w", samplerate=noisy.samplerate, channels=noisy.channels
            )
        if outputfile:
            outputfile.write(noisy_data)
        now = time.time()
        perc = int(100 * (now - start) / seconds)
        if perc <= 100:
            print(f"{perc}%")

    if outputfile:
        outputfile.close()
    labels = ["samples", "time", "correlation1", "correlation2"]
    result = pd.DataFrame.from_records(delays, columns=labels, coerce_float=True)

    output_filename = outputcsv
    if (output_filename is not None) & (output_filename[-4:] != ".csv"):
        output_filename = output_filename + ".csv"
    if output_filename:
        result.to_csv(output_filename, index=False)
    if failed_counter > 0:
        total = len(delays) + failed_counter
        print(
            f"Failed {failed_counter} tries out of {total} "
            f"({int(100 * failed_counter/(total))})%"
        )
        print(
            "If this percentage is high it is an indicator on problems "
            "with the transmission."
        )

    if len(result > 0):
        print(f'Shortest latency: {int(np.min(result["time"]))} ms')
        print(f'Longest latency: {int(np.max(result["time"]))} ms')
        print(f'Average latency: {int(np.mean(result["time"]))} ms')

    os.remove(tmpfile)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i", "--impulse", required=True, help="path to a (preferable) short pcm file"
    )
    parser.add_argument(
        "--source",
        required=False,
        default=None,
        help="If set, calculation will be done on this file",
    )
    parser.add_argument(
        "-o", "--output_csv", required=True, help="Write the result to this csv file"
    )
    parser.add_argument(
        "-t",
        "--seconds",
        required=False,
        default=30,
        type=int,
        help="Length of the test in secs",
    )
    parser.add_argument(
        "-d",
        "--delay",
        required=False,
        default=3,
        type=int,
        help="Distance between signal in secs",
    )
    parser.add_argument(
        "-s",
        "--save",
        required=False,
        default="",
        help="Save and concatenate the recorded audio",
    )
    parser.add_argument("-v", "--verbose", required=False, action="store_true")
    parser.add_argument(
        "-tr",
        "--threshold",
        required=False,
        type=int,
        default=25,
        help="Threshold for considering a hit. 100 is perfect match",
    )
    options = parser.parse_args()
    global running
    global index

    threshold = options.threshold
    if options.source is not None:
        delay = findDelayInFile(
            options.source, options.impulse, threshold, options.verbose
        )
        if delay is not None:
            print(
                f"The delay is \n{delay[0]} samples - {delay[1]} ms - correlations:({delay[2]}/{delay[3]})"
            )
        else:
            print("Failed to find an echo")
    else:
        measureDelay(
            options.impulse,
            threshold,
            options.delay,
            options.seconds,
            options.output_csv,
            options.save,
            options.verbose,
        )


if __name__ == "__main__":
    main()
