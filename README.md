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


## feedbackcount.py
This tool uses the method of measuring distances between feedback signals.
One of the issues with the findecho.py method is that sometimes feedback is
hard to get a round. With a realtive short distance between the microphone
and the speaker and/or relfective surfaces, frequencies from the speaker signal
are being picket up and amplified by the microphone.
If there is a delay in the system this will lead to a pulsating system and the
idea here is that the frequency of the pulses is directly correlated to the delay
time.
The system will do a recording (with or without providing a stimuli to start the
feedback) and analyse the spectrum, pick out a strong signal and bandpass filter
around it. That signal is then filtered further and the the peaks are identified
and the distance between them is counted and put in a histogram where the highest
bin will determine the delay.

One of the complexities with the feedback analysis is that it is possible that a
multitude of signals are feeding simultanously and the script needs to separate them.
The idea is that all related peaks which are a function of the real delay will have
the same periodicity and the histogram will peak for that period.
To measure it is probably best to have a system that does not feedback by itself but
needs a stimuli to start the loop. Also, limiting the number of stimuli signals makes
is easier to not have them confusing the measurements.

usage: feedbackcount.py [-h] [-i IMPULSE] [--source SOURCE]
                        -o OUTPUT_CSV [-t SECONDS] [-d DELAY]
                        [-s SAVE] [-v] [-mi MINTIMEMS] [-mx MAXTIMEMS] [-nf]

optional arguments:
  -h, --help            show this help message and exit
  -i IMPULSE, --impulse IMPULSE
  --source SOURCE       If set, calculation will be done on this file
  -o OUTPUT_CSV, --output_csv OUTPUT_CSV
  -t SECONDS, --seconds SECONDS
  -d DELAY, --delay DELAY
  -s SAVE, --save SAVE  Save and concatenate the recorded audio
  -v, --verbose
  -mi MINTIMEMS, --mintimems MINTIMEMS
                        Shortes pulse to consider in feedback analysis
  -mx MAXTIMEMS, --maxtimems MAXTIMEMS
                        Shortes pulse to consider in feedback analysis
  -nf, --nofilter       Do not filter for frequency peaks


Example:
>python3.8 python/feedbackcount.py -i audio/chirp_300-3k.wav -o pixel3 -t 30 -d 10

The command above will start a measurement using the '-i' as stimuli running for 30 s
and the stimuli to be player every 10th seconds.

* The bandpass filter can be skipped and limits on the shortest/longest delay to look for
can be set. In noisy environment slow periodict signals like air condition can easily be
misstaken for the real signal. To see what signal is analyzed, use the vebose option.
* Using the '--source' a prerecorded signal containing a feedback is analyzed.
* If measuring a system already in feedback condition skip the impulse and just record.
