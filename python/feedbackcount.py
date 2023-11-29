#!/usr/bin/env pythonx
# (c) Facebook, Inc. and its affiliates.

import argparse
import os
import time

import numpy as np
import numpy.ma as ma
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
    Filter out weak peaks to reduce noise.

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
    hmean = np.mean(data[peaks])
    # filter everything below half mean
    mask = ma.masked_where(data[peaks] < hmean / 10, data[peaks])
    filtered = np.ma.masked_where(np.ma.getmask(mask), peaks)
    compr = ma.compressed(filtered)
    if verbose:
        fig, ax = plt.subplots()
        ax.plot(data)
        ax.plot(peaks, data[peaks])
        ax.plot(filtered, data[filtered], "o", label="Filtered")
        ax.set_yscale("log")
        ax.set_ylabel("magnitude")
        ax.set_xlabel("time (samples)")
        fig.suptitle("Find peaks")
        plt.legend()
        plt.show()
    return compr


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
    filt = list(filter(lambda diff: diff > minsamples and diff < maxsamples, diffs))
    if len(filt) > 0:
        average = int(np.mean(filt))
        try:
            rounded = round(1000 * average / samplerate)
        except Exception:
            return None
        return average, rounded
    else:
        return None


def findMostCommonDiff(data, samplerate, minimum_ms, maximum_ms, verbose):
    """Calculate distance between peaks in data

    intpuArgst:
        data: input data
        samplerate: samplerate of data
        minimum_ms: shortest time period to consider

    Returns:
        average: average distance in samples
        average time: average distance in time in ms
        count: the number of repetitions found
    """

    diffs = []
    length = int(len(data) / 2)
    if length == 0:
        length = 1
    for offset in range(1, length + 1):
        for index in range(offset, len(data)):
            diffs.append(int((data[index] - data[index - offset])))

    if len(diffs) < 5:
        if len(diffs) > 0:
            average = int(np.round(np.mean(diffs)))
            rounded = round(1000 * average / samplerate)
            return average, rounded, 1
        else:
            return None
    else:
        minsamples = int(samplerate * (minimum_ms / 1000))
        maxsamples = int(samplerate * (maximum_ms / 1000))
        hist, edges = np.histogram(
            diffs, int(samplerate / 1000), range=(minsamples, maxsamples)
        )
        maxindex = np.argmax(hist)
        order = np.argsort(hist)[::-1]
        maxval = edges[order[0]]
        secmaxval = edges[order[1]]
        ratio = maxval / secmaxval
        # if the ratio is close to 2 lets use the shorter interval

        if abs(ratio - 2) < 0.1:
            if maxval > secmaxval:
                maxindex = order[1]
        inbin = list(
            filter(
                lambda diff: diff >= edges[maxindex] and diff < edges[maxindex + 1],
                diffs,
            )
        )
        average = int(np.round(np.mean(inbin)))
        if verbose:
            fig, ax = plt.subplots()
            ax.plot(edges[:-1], hist)
            ax.set_xlabel("time (samples)")
            ax.set_ylabel("count")
            ax.grid()
            fig.suptitle("Peak distances histogram")
            plt.show()
        if average > 0:
            try:
                rounded = round(1000 * average / samplerate)
            except Exception:
                return None
            return average, rounded, hist[maxindex]
        else:
            return None


def bpFilterSignal(data, hp, lp, order, samplerate, verbose):
    """Band pass filter the signal

    Bandpass with butterworth filter with hp and lp

    intpuArgst:
        data: input data
        hp: -3dB frequency for high pass
        lp: -3dB frequency for low pass
        samplerate: samplerate of data

    Returns:
        data: filtered version of data
    """

    n = len(data)
    t = 1.0 / samplerate
    if hp is None:
        hp = 0
    if lp is None:
        lp = int(samplerate / 2)
    if verbose:
        fig, (ax1, ax2) = plt.subplots(2)
        yf = fft(data)
        xf = fftfreq(n, t)
        peak = np.argmax(np.abs(yf))
        peakf = int(np.abs(xf[peak]))
        ax1.plot(xf[: int(n / 2)], 1.0 / n * np.abs(yf[: int(n / 2)]))
        ax1.plot(xf[peak], 1.0 / n * np.abs(yf[peak]), "o", color="red")
        ax1.vlines(x=lp, ymax=1.0 / n * max(np.abs(yf)), ymin=0, color="green")
        ax1.vlines(x=hp, ymax=1.0 / n * max(np.abs(yf)), ymin=0, color="green")
        ax1.grid()
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("magnitude")
        fig.suptitle(f"FFT non filter vs filtered peak {peakf} Hz")
        print(f"peak: {peakf} Hz")

    sos = signal.butter(order, [hp, lp], "band", fs=samplerate, output="sos")
    data = signal.sosfilt(sos, data)
    if verbose:
        yf = fft(data)
        xf = fftfreq(n, t)
        prevPeakf = peakf
        peak = np.argmax(np.abs(yf))
        peakf = int(np.abs(xf[peak]))
        ax2.plot(xf[: int(n / 2)], 1.0 / n * np.abs(yf[: int(n / 2)]))
        ax2.plot(xf[peak], 1.0 / n * np.abs(yf[peak]), "o", color="red")
        ax2.vlines(x=lp, ymax=1.0 / n * max(np.abs(yf)), ymin=0, color="green")
        ax2.vlines(x=hp, ymax=1.0 / n * max(np.abs(yf)), ymin=0, color="green")
        ax2.grid()
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("magnitude")
        print(f"new peak: {peakf} Hz")
        fig.suptitle(
            f"FFT non filter vs filteres, original peak "
            f"{prevPeakf} Hz, filtered {peakf} Hz"
        )
        plt.show()
    return data


def findPeak(data, samplerate, verbose):
    n = len(data)
    t = 1.0 / samplerate
    yf = fft(data)
    xf = fftfreq(n, t)
    peak = np.argmax(np.abs(yf))
    peakf = int(np.abs(xf[peak]))

    return peakf


def findDelayUsingFeedbackFreqInFile(
    noisy_path,
    min_length_ms,
    max_length_ms,
    nofilter=False,
    hp=None,
    lp=None,
    verbose=False,
):
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
        data, noisy.samplerate, min_length_ms, max_length_ms, nofilter, hp, lp, verbose
    )


def findDelayUsingFeedbackFreqInData(
    data,
    samplerate,
    min_length_ms,
    max_length_ms,
    nofilter=False,
    hp=None,
    lp=None,
    verbose=False,
):
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
    if hp is not None or lp is not None:
        data = bpFilterSignal(data, hp, lp, 4, samplerate, verbose)
    elif nofilter is False:
        # filter out lf noise (ac, street etc) and hf
        # 1920 - 10k Hz for 48k
        # since low f`requencies contain more energy
        # it is good to roll off a little higher
        low = int(samplerate / 25)
        high = int(samplerate / 4)
        data = bpFilterSignal(data, low, high, 2, samplerate, verbose)

    if nofilter is False:
        peakf = findPeak(data, samplerate, verbose)
        w = peakf / (40)
        low = peakf - w
        high = peakf + w
        data = bpFilterSignal(data, low, high, 4, samplerate, verbose)
    window_len = int(0.1 * samplerate)  # 100ms
    data = data * data
    s = np.r_[data[window_len - 1 : 0 : -1], data, data[-2 : -window_len - 1 : -1]]
    w = np.hanning(window_len)
    sdata = np.convolve(w / w.sum(), s, mode="valid")
    peaks = findPeaks(sdata, verbose)
    # return calcDiff(peaks, samplerate, min_length_ms)
    return findMostCommonDiff(peaks, samplerate, min_length_ms, max_length_ms, verbose)


def measureDelayUsingFeedback(
    impulse,
    delay,
    seconds,
    outputcsv,
    save,
    verbose=False,
    min_length_ms=200,
    max_length_ms=600,
    nofilter=False,
    hp=None,
    lp=None,
):
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
        data = findDelayUsingFeedbackFreqInData(
            noisy_data,
            noisy.samplerate,
            min_length_ms,
            max_length_ms,
            nofilter,
            hp,
            lp,
            verbose,
        )
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

        if data is not None:
            if data[2] < 20:
                print(f"Number of repetition is {data[2]}.\n" "Consider a longer test")
        else:
            print("No repetitions found - check setup.")
    if outputfile:
        outputfile.close()
    labels = ["samples", "time", "count"]
    result = pd.DataFrame.from_records(delays, columns=labels, coerce_float=True)

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
            "\nIf this percentage is high it is an indicator "
            "on problems with the transmission.\n"
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
        "-mi",
        "--mintimems",
        required=False,
        default=150,
        type=int,
        help="Shortes pulse to consider in feedback analysis",
    )
    parser.add_argument(
        "-mx",
        "--maxtimems",
        required=False,
        default=600,
        type=int,
        help="Shortes pulse to consider in feedback analysis",
    )
    parser.add_argument(
        "-nf",
        "--nofilter",
        action="store_true",
        help="Do not filter for frequency peaks",
    )
    parser.add_argument(
        "-hp",
        "--hpfreq",
        default=None,
        type=int,
        help="High pass",
    )
    parser.add_argument(
        "-lp",
        "--lpfreq",
        default=None,
        type=int,
        help="Low pass",
    )
    options = parser.parse_args()
    global running
    global index
    if options.source is not None:
        delay, delay_ms, count = findDelayUsingFeedbackFreqInFile(
            options.source,
            options.mintimems,
            options.maxtimems,
            options.nofilter,
            options.hpfreq,
            options.lpfreq,
            options.verbose,
        )
        if delay is not None:
            print(f"The delay is \n{delay} samples - {delay_ms} ms")
            if count < 20:
                print(f"Number of repetition is {count}.\n" "Consider a longer test")
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
            options.mintimems,
            options.maxtimems,
            nofilter=options.nofilter,
        )


if __name__ == "__main__":
    main()
