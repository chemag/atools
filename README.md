# atools: A set of audio tools

## findecho.py
This tool will play a short sample and record simultanously looking for the first duplication of the played audio.
Typical use case is to record roundtrip delay on a single device or end to end delay in a communication system between devices.
The audio directory contains a 8kHz chirop which can be used as a signal.

usage: findecho.py [-h] -i IMPULSE -o OUTPUT_CSV [-t SECONDS] [-d DELAY]
                   [-s SAVE] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -i IMPULSE, --impulse IMPULSE
  -o OUTPUT_CSV, --output_csv OUTPUT_CSV
  -t SECONDS, --seconds SECONDS
  -d DELAY, --delay DELAY
  -s SAVE, --save SAVE
  -v, --verbose

Example:
> python3 python/findecho.py -i audio/chirp_300-3k.wav -o pixel3 -t 30
