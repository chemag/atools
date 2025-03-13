#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates.

from math import log10

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import subprocess
import sys

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


def run(command, **kwargs):
    debug = kwargs.get("debug", 0)
    dry_run = kwargs.get("dry_run", False)
    env = kwargs.get("env", None)
    stdin = subprocess.PIPE if kwargs.get("stdin", False) else None
    bufsize = kwargs.get("bufsize", 0)
    universal_newlines = kwargs.get("universal_newlines", False)
    default_close_fds = True if sys.platform == "linux2" else False
    close_fds = kwargs.get("close_fds", default_close_fds)
    shell = kwargs.get("shell", type(command) in (type(""), type("")))
    logfd = kwargs.get("logfd", sys.stdout)
    if debug > 0:
        print(f"$ {command}", file=logfd)
    if dry_run:
        return 0, b"stdout", b"stderr"
    gnu_time = kwargs.get("gnu_time", False)
    if gnu_time:
        # GNU /usr/bin/time support
        command = f"/usr/bin/time -v {command}"

    p = subprocess.Popen(  # noqa: E501
        command,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=bufsize,
        universal_newlines=universal_newlines,
        env=env,
        close_fds=close_fds,
        shell=shell,
    )
    # wait for the command to terminate
    if stdin is not None:
        out, err = p.communicate(stdin)
    else:
        out, err = p.communicate()
    returncode = p.returncode
    # clean up
    del p
    if gnu_time:
        # make sure the stats are there
        GNU_TIME_BYTES = b"\n\tUser time"
        assert GNU_TIME_BYTES in err, "error: cannot find GNU time info in stderr"
        gnu_time_str = err[err.index(GNU_TIME_BYTES) :].decode("ascii")
        gnu_time_stats = gnu_time_parse(gnu_time_str, logfd, debug)
        err = err[0 : err.index(GNU_TIME_BYTES) :]
        return returncode, out, err, gnu_time_stats
    # return results
    return returncode, out, err


GNU_TIME_DEFAULT_KEY_DICT = {
    "Command being timed": "command",
    "User time (seconds)": "usertime",
    "System time (seconds)": "systemtime",
    "Percent of CPU this job got": "cpu",
    "Elapsed (wall clock) time (h:mm:ss or m:ss)": "elapsed",
    "Average shared text size (kbytes)": "avgtext",
    "Average unshared data size (kbytes)": "avgdata",
    "Average stack size (kbytes)": "avgstack",
    "Average total size (kbytes)": "avgtotal",
    "Maximum resident set size (kbytes)": "maxrss",
    "Average resident set size (kbytes)": "avgrss",
    "Major (requiring I/O) page faults": "major_pagefaults",
    "Minor (reclaiming a frame) page faults": "minor_pagefaults",
    "Voluntary context switches": "voluntaryswitches",
    "Involuntary context switches": "involuntaryswitches",
    "Swaps": "swaps",
    "File system inputs": "fileinputs",
    "File system outputs": "fileoutputs",
    "Socket messages sent": "socketsend",
    "Socket messages received": "socketrecv",
    "Signals delivered": "signals",
    "Page size (bytes)": "page_size",
    "Exit status": "status",
}


GNU_TIME_DEFAULT_VAL_TYPE = {
    "int": [
        "avgtext",
        "avgdata",
        "avgstack",
        "avgtotal",
        "maxrss",
        "avgrss",
        "major_pagefaults",
        "minor_pagefaults",
        "voluntaryswitches",
        "involuntaryswitches",
        "swaps",
        "fileinputs",
        "fileoutputs",
        "socketsend",
        "socketrecv",
        "signals",
        "page_size",
        "status",
        "usersystemtime",
    ],
    "float": [
        "usertime",
        "systemtime",
    ],
    "timedelta": [
        "elapsed",
    ],
    "percent": [
        "cpu",
    ],
}


def gnu_time_parse(gnu_time_str, logfd, debug):
    gnu_time_stats = {}
    for line in gnu_time_str.split("\n"):
        if not line:
            # empty line
            continue
        # check if we know the line
        line = line.strip()
        for key1, key2 in GNU_TIME_DEFAULT_KEY_DICT.items():
            if line.startswith(key1):
                break
        else:
            # unknown key
            print(f"warn: unknown gnutime line: {line}", file=logfd)
            continue
        val = line[len(key1) + 1 :].strip()
        # fix val type
        if key2 in GNU_TIME_DEFAULT_VAL_TYPE["int"]:
            val = int(val)
        elif key2 in GNU_TIME_DEFAULT_VAL_TYPE["float"]:
            val = float(val)
        elif key2 in GNU_TIME_DEFAULT_VAL_TYPE["percent"]:
            val = float(val[:-1])
        elif key2 in GNU_TIME_DEFAULT_VAL_TYPE["timedelta"]:
            # '0:00.02'
            timedelta_re = r"((?P<min>\d+):(?P<sec>\d+).(?P<centisec>\d+))"
            res = re.search(timedelta_re, val)
            timedelta_sec = int(res.group("min")) * 60 + int(res.group("sec"))
            timedelta_centisec = int(res.group("centisec"))
            timedelta_sec += timedelta_centisec / 100.0
            val = timedelta_sec
        gnu_time_stats[key2] = val
    gnu_time_stats["usersystemtime"] = (
        gnu_time_stats["usertime"] + gnu_time_stats["systemtime"]
    )
    return gnu_time_stats
