#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates.

import argparse
import os
import time

import common as cm
import numpy as np
import pandas as pd
import play_rec as pr
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--impulse', required=True)
    parser.add_argument('-o', '--output_csv', required=True)
    parser.add_argument('-t', '--seconds', required=False, default=30,
                        type=int)
    parser.add_argument('-d', '--delay', required=False, default=2, type=int)
    parser.add_argument('-s', '--save', required=False, default='')
    parser.add_argument('-v', '--verbose', required=False,
                        action='store_true')
    options = parser.parse_args()
    global running
    global index

    start = time.time()
    now = start
    if options.impulse[-3:] == 'wav':
        ref = sf.SoundFile(options.impulse, 'r')
        ref_data = ref.read()
    else:
        print('Only wav file supported')
        exit(0)

    outputfile = None
    threshold = 25
    failed_counter = 0
    delays = []

    while now - start < options.seconds:
        pr.play_and_record(options.impulse, options.output_csv + '_.wav',
                           options.delay)
        noisy = sf.SoundFile(options.output_csv + '_.wav', 'r')
        noisy_data = noisy.read()
        if (len(options.save) > 0) & (outputfile is None):
            outputfile = sf.SoundFile(
                options.save, 'w', samplerate=noisy.samplerate,
                channels=noisy.channels)

        data = cm.find_markers(
            ref_data, noisy_data, threshold, noisy.samplerate,
            verbose=options.verbose)
        if len(data) > 1:
            diff = data.loc[1, 'sample'] - data.loc[0, 'sample']
            delays.append(
                [
                    diff,
                    round(diff / noisy.samplerate * 1000),
                    data.loc[0, 'correlation'],
                    data.loc[1, 'correlation'],
                ]
            )
            if outputfile:
                outputfile.write(noisy_data)
        else:
            failed_counter += 1

        now = time.time()
        perc = int(100 * (now - start) / options.seconds)
        if perc <= 100:
            if failed_counter > 0:
                total = len(delays) + failed_counter
                print(f'{perc}% ({int(100 * failed_counter/(total))}%'
                      ' failures)')
            else:
                print(f'{perc}%')

    if outputfile:
        outputfile.close()
    labels = ['samples', 'time', 'correlation1', 'correlation2']
    result = pd.DataFrame.from_records(delays, columns=labels,
                                       coerce_float=True)

    output_filename = options.output_csv
    if (options.output_csv is not None) & (options.output_csv[-4:] != '.csv'):
        output_filename = options.output_csv + '.csv'
    if options.output_csv:
        result.to_csv(output_filename, index=False)
    if failed_counter > 0:
        total = len(delays) + failed_counter
        print(
            f'Failed {failed_counter} tries out of {total} '
            f'({int(100 * failed_counter/(total))})%')
        print(
            'If this percentage is high it is an indicator on problems '
            'with the transmission.')

    if len(result > 0):
        print(f'Shortest latency: {int(np.min(result["time"]))} ms')
        print(f'Longest latency: {int(np.max(result["time"]))} ms')
        print(f'Average latency: {int(np.mean(result["time"]))} ms')

    os.remove(options.output_csv + '_.wav')


if __name__ == '__main__':
    main()
