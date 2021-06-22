#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates.

import argparse

import numpy as np
import scipy.io.wavfile as wf
import sounddevice as sd

running = True


def record(output, duration_sec, samplerate):
    fs = samplerate
    global rec_data
    global inputfile
    global last_start
    global data
    global index

    rec_data = np.empty((0, 1), np.int16)

    def callback(indata, outdata, frames, time, status):
        global rec_data
        global index
        global last_start
        global data

        rec_data = np.append(rec_data, indata[::])

    with sd.Stream(samplerate=fs, channels=1, callback=callback, dtype=np.int16):
        sd.sleep(duration_sec * 1000)
    wf.write(output, fs, rec_data)  # Save as WAV file


def play_and_record(signal, output, duration_sec):
    global rec_data
    global inputfile
    global last_start
    global data
    global index
    signalfile = signal
    rec_data = np.empty((0, 1), np.int16)
    index = -1
    last_start = -100
    fs, data = wf.read(signalfile)

    def callback(indata, outdata, frames, time, status):
        global rec_data
        global inputfile
        global index
        global last_start
        global data

        length = len(data)
        ct = int(time.currentTime)
        rec_data = np.append(rec_data, indata[::])

        if (ct - last_start > 10) & (index < 0):
            index = 0
            last_start = ct
        if index >= 0:
            left = length - index

            if left > frames:
                outdata[:, 0] = data[index : index + frames]
                index += frames
            else:
                index = -1
                outdata[:, 0] = np.zeros(frames)
        else:
            outdata[:, 0] = np.zeros(frames)

    with sd.Stream(samplerate=fs, channels=1, callback=callback, dtype=np.int16):
        sd.sleep(duration_sec * 1000)
    wf.write(output, fs, rec_data)  # Save as WAV file


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-t", "--seconds", required=False, default=30)
    options = parser.parse_args()
    global running
    global index
    play_and_record(options.input, options.output, options.seconds, 16000)


if __name__ == "__main__":
    main()
