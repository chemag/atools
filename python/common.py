#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates.

from math import log10

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

global options


def floatToDB(val):
    """
    Calculates the dB values from a floating point representation
    ranging between -1.0 to 1.0 where 1.0 is 0 dB
    """
    if val <= 0:
        return -100.0
    else:
        return 20.0 * log10(val)


def dBToFloat(val):
    """
    Calculates a float value ranging from -1.0 to 1.0
    Where 1.0 is 0dB
    """
    return 10 ** (val / 20.0)


def audio_levels(audiofile, start=0, end=-1):
    """
    Calculates rms and max peak level in dB
    """

    blocksize = audiofile.channels * audiofile.samplerate * 10
    peak_level = [0] * audiofile.channels
    rms = [0] * audiofile.channels
    peak = [0] * audiofile.channels
    total_level = [0] * audiofile.channels
    crest = [0] * audiofile.channels
    bias = [0] * audiofile.channels
    block_counter = 0
    audiofile.seek(start)

    while audiofile.tell() < audiofile.frames:
        data = audiofile.read(blocksize)
        for channel in range(0, audiofile.channels):
            if audiofile.channels == 1:
                data_ = data
            else:
                data_ = data[:, channel]
            total_level[channel] += np.sum(data_)
            rms[channel] += np.mean(np.square(data_))
            peak[channel] = max(abs(data_))
            if peak[channel] > peak_level[channel]:
                peak_level[channel] = peak[channel]
        block_counter += 1

    for channel in range(0, audiofile.channels):
        rms[channel] = np.sqrt(rms[channel] / block_counter)
        crest[channel] = round(floatToDB(peak_level[channel] / rms[channel]), 2)
        bias[channel] = round(
            floatToDB(
                total_level[channel] / (block_counter * 10 * audiofile.samplerate)
            ),
            2,
        )
        rms[channel] = round(floatToDB(rms[channel]), 2)
        peak_level[channel] = round(floatToDB(peak_level[channel]), 2)

    return rms, peak_level, crest, bias


def visualize_corr(data, ref_data, corr):
    fig, (ax_data, ax_ref_data, ax_corr) = plt.subplots(3, 1, figsize=(4.8, 4.8))
    ax_data.plot(ref_data)
    ax_data.set_title("Ref impulse")
    ax_data.set_xlabel("Sample Number")
    ax_ref_data.plot(data)
    ax_ref_data.set_title("Signal with noise")
    ax_ref_data.set_xlabel("Sample Number")
    ax_corr.plot(corr)
    ax_corr.set_title("Cross-correlated signal")
    ax_corr.set_xlabel("Lag")
    ax_data.margins(0, 0.1)
    ax_ref_data.margins(0, 0.1)
    ax_corr.margins(0, 0.1)
    fig.tight_layout()
    plt.show()


def match_buffers(data, ref_data, gain=0, verbose=False):
    """
    Tries to find ref_data in data using correlation measurement.
    """
    global options
    size = len(ref_data)

    if gain != 0:
        data = np.multiply(data, gain)
    corr = np.correlate(data, ref_data)
    val = max(corr)

    index = np.where(corr == val)[0][0]

    cc_ = np.corrcoef(data[index : index + size], ref_data)[1, 0] * 100
    if np.isnan(cc_):
        cc_ = 0
    cc = int(cc_ + 0.5)
    if verbose:
        print(f"{val} @ {index}, cc = {cc}")
        visualize_corr(data, ref_data, corr)
    return index, cc


def find_markers(reference, noisy, threshold, samplerate, verbose=False):
    # Sets how close we can find multiple matches, 100ms
    window = int(0.1 * samplerate)
    max_pos = 0

    silence = np.full((len(reference)), 0)
    noisy = np.append(noisy, silence)
    read_len = int(len(reference) + window)
    ref_duration = len(reference) / samplerate
    counter = 0
    last = 0
    split_times = []
    while last <= len(noisy) - len(reference):
        index, cc = match_buffers(
            noisy[last : last + read_len], reference, verbose=verbose
        )

        index += last
        pos = index - max_pos
        if pos < 0:
            pos = 0
        time = pos / samplerate
        if cc > threshold:
            if (len(split_times) > 0) and (
                abs(time - split_times[-1][1]) < ref_duration / 2
            ):
                if split_times[-1][2] <= cc:
                    split_times.pop(-1)
            else:
                split_times.append([pos, time, cc])

        last += window
        counter += 1

    data = pd.DataFrame()
    labels = ["sample", "time", "correlation"]
    data = pd.DataFrame.from_records(split_times, columns=labels, coerce_float=True)

    return data
