#!/usr/bin/env pythonx
# (c) Facebook, Inc. and its affiliates.

import argparse
import os
import time

import numpy as np
import pandas as pd
import play_rec as pr
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal


def findPeaks(data, verbose=False):
    """Finds peaks in the data

    A peak is defined as a list of three times
    where the middle one is higher than the other two

    intpuArgst:
        data: input data

    Returns:
        peaks: a list of peak index in data
    """

    peaks = []
    last = 0
    up = False
    counter = -1
    for val in data:
        if val <= last and up:
            # peak?
            peaks.append(counter)
            up = False
        elif val > last:
            up = True
        last = val
        counter += 1
    if verbose:
        plt.plot(data)
        plt.plot(peaks, data[peaks])
        plt.show()
    return peaks


def calcDiff(data, samplerate, minimum_ms):
    """Calculate distance between peaks in data

    intpuArgst:
        data: input data
        samplerate: samplerate of data
        minimum_ms: shortest time period to consider

    Returns:
        average: average distance in samples
        average time: average distance in time in ms
    """

    diffs = []
    last = 0
    for val in data:
        if last != 0:
            diffs.append(int((val - last)))
        last = val

    minsamples = int(samplerate * (minimum_ms / 1000))
    maxsamples = samplerate  # 1 sec
    filt = list(filter(lambda diff: diff > minsamples and diff < maxsamples,
                       diffs))
    if len(filt) > 0:
        average = int(np.mean(filt))
        try:
            rounded = round(1000 * average / samplerate)
        except Exception:
            return None
        return average, rounded
    else:
        return None


def findFreqPeaksAndFilter(data, samplerate, verbose):
    """Find strongest frequency peak and filters

    Find the strongest frequency and funs a bandpass
    filter between half and double the frequency
    intpuArgst:
        data: input data
        samplerate: samplerate of data

    Returns:
        data: filtered version of data
    """

    n = len(data)
    t = 1.0 / samplerate
    yf = fft(data)
    xf = fftfreq(n, t)
    peak = np.argmax(np.abs(yf))
    peakf = int(np.abs(xf[peak]))

    if verbose:
        print(f'peak: {peakf} Hz')
        plt.plot(xf, 1.0 / n * np.abs(yf))

    low = peakf / 2
    high = peakf * 2

    if verbose:
        print(f'bp: {low} - {high}')

    sos = signal.butter(2, [low, high], 'band', fs=samplerate, output='sos')
    filtered = signal.sosfilt(sos, data)
    yf = fft(filtered)
    if verbose:
        plt.plot(xf, 1.0 / n * np.abs(yf))
        plt.grid()
        plt.show()

    return filtered


def findDelayUsingFeedbackFreqInFile(
        noisy_path, min_length_ms, nofilter=False, verbose=False):
    """Find first two signals in an audio file

    Calculates the difference between the first two signals
    matching reference to sufficient degree.
    intpuArgst:
        noisy_path: path to wavfile to be searched
        min_length_ms: shortes peak distance to look for

    Returns:
        delay: in samplse,
        delay: in ms,
    """

    noisy = sf.SoundFile(noisy_path, "r")
    data = noisy.read()
    return findDelayUsingFeedbackFreqInData(
        data, noisy.samplerate, min_length_ms, nofilter, verbose)


def findDelayUsingFeedbackFreqInData(
        data, samplerate, min_length_ms, nofilter=False, verbose=False):
    """Find first two signals in an audio file

    Calculates the difference between the first two signals
    matching reference to sufficient degree.
    intpuArgst:
        noisy_path: path to wavfile to be searched
        min_length_ms: shortes peak distance to look for

    Returns:
        delay: in samplse,
        delay: in ms,
    """
    if nofilter is False:
        data = findFreqPeaksAndFilter(data, samplerate, verbose)
    window_len = int(0.1 * samplerate)  # 100ms
    data = data * data
    s = np.r_[data[window_len - 1:0:-1], data, data[-2:-window_len - 1:-1]]
    w = np.hanning(window_len)
    sdata = np.convolve(w / w.sum(), s, mode='valid')
    peaks = findPeaks(sdata, verbose)
    return calcDiff(peaks, samplerate, min_length_ms)


def measureDelayUsingFeedback(impulse,
                              delay,
                              seconds,
                              outputcsv,
                              save,
                              verbose=False,
                              minlenghtms=200,
                              nofilter=False):
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

    outputfile = None
    failed_counter = 0
    delays = []
    start = time.time()
    now = start
    tmpfile = outputcsv + "_.wav"
    while now - start < seconds:
        pr.play_and_record(impulse, tmpfile, delay)
        noisy = sf.SoundFile(tmpfile, "r")
        noisy_data = noisy.read()
        data = findDelayUsingFeedbackFreqInData(noisy_data,
                                                noisy.samplerate,
                                                minlenghtms,
                                                nofilter,
                                                verbose)
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
    labels = ["samples", "time"]
    result = pd.DataFrame.from_records(
        delays, columns=labels, coerce_float=True)

    output_filename = outputcsv
    if (output_filename is not None) & (output_filename[-4:] != ".csv"):
        output_filename = output_filename + ".csv"
    if output_filename:
        result.to_csv(output_filename, index=False)
    if failed_counter > 0:
        total = len(delays) + failed_counter
        print(
            f"Failed {failed_counter} tries out of "
            f"{total} ({int(100 * failed_counter/(total))})%"
        )
        print(
            "If this percentage is high it is an indicator "
            "on problems with the transmission."
        )

    if len(result > 0):
        print(f"Shortest latency: {int(np.min(result['time']))} ms")
        print(f"Longest latency: {int(np.max(result['time']))} ms")
        print(f"Average latency: {int(np.mean(result['time']))} ms")

    os.remove(tmpfile)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i",
        "--impulse",
        required=False,
        help="path to a (preferable) short pcm file",
    )
    parser.add_argument(
        "--source",
        required=False,
        default=None,
        help="If set, calculation will be done on this file",
    )
    parser.add_argument(
        "-o", "--output_csv",
        required=True,
        help="Write the result to this csv file"
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
        "-ms",
        "--timems",
        required=False,
        default=150,
        type=int,
        help="Shortes pulse to consider in feedback analysis",
    )
    parser.add_argument(
        "-nf",
        "--nofilter",
        action='store_true',
        help="Do not filter for frequency peaks",
    )
    options = parser.parse_args()
    global running
    global index
    if options.source is not None:
        delay = findDelayUsingFeedbackFreqInFile(options.source,
                                                 options.timems,
                                                 options.nofilter,
                                                 options.verbose)
        if delay is not None:
            print(
                f"The delay is \n{delay[0]} samples - {delay[1]} ms"
            )
        else:
            print("Failed to find an echo")
    else:
        measureDelayUsingFeedback(
            options.impulse,
            options.delay,
            options.seconds,
            options.output_csv,
            options.save,
            options.verbose,
            nofilter=options.nofilter
        )


if __name__ == "__main__":
    main()
