#!/usr/bin/env python

from __future__ import print_function
from optparse import OptionParser
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import subprocess
import os
import json, codecs
import struct 

cwd = os.getcwd()
args = OptionParser()
args.add_option('-i', '--input', dest='filename')
args.add_option('-o', '--output', dest='dir', default="data")
args.add_option('-v', '--interval', dest='int', default=1, type='float')
args.add_option('-s', '--spread', dest='spread', default=24, type='float')
args.add_option('-b', '--begin', dest='begin', default=-12, type='float')
args.add_option('-r', '--rate', dest='rate', default=48000.0, type='float')
opts, args = args.parse_args()
file = os.path.basename(opts.filename)

y, sr = librosa.load(opts.filename, sr=opts.rate, mono=True)
sampleRate = opts.rate

data = {}
data['files'] = []

def cut(y, offsets):
  for i in np.arange(len(offsets)):
    j = 0 if i < 1 else offsets[i-1]
    y_slice = slice(y, offset=j, limit=offsets[i]) 
    #x_slice = np.linspace(j / float(sr), offsets[i] / float(sr), y_slice.size) 
    #librosa.output.write_wav(opts.dir + '/cut-' + str(i).zfill(4) + '.wav', y=y_slice, sr=sr)
    #xy = np.empty(y_slice.size + x_slice.size, dtype=x_slice.dtype)
    #xy[0::2] = x_slice
    #xy[1::2] = y_slice
    #print(xy)
    file = opts.dir + '/cut-' + str(i).zfill(4) + '.raw'
    data['files'].append(file)
    open(file, 'w').write(struct.pack('f'*y_slice.size, *y_slice)) 

def slice(y, offset=0, limit=None):
  return y[offset:(limit if limit is not None else None)]


''' 
  #if split harmonic and percussive, probably not necessary
  y_harmonic, y_percussive = librosa.effects.hpss(y, kernel_size=[256, 1024])

  librosa.output.write_wav(opts.dir + '/harmonic.wav', y=y_harmonic, sr=sr)
  librosa.output.write_wav(opts.dir + '/percussive.wav', y=y_percussive, sr=sr)

  perc_onset = librosa.onset.onset_strength(y_percussive, sr=sr, aggregate=np.sum)
  tempo_perc, beat_frames_perc = librosa.beat.beat_track(onset_envelope=perc_onset, sr=sr)
  beat_frames_perc = librosa.core.frames_to_samples(beat_frames_perc)

  percussive_onset_times = librosa.frames_to_time(np.arange(len(beat_frames_perc)), sr=sr)

  harmonic_onset = librosa.onset.onset_strength(y_harmonic, sr=sr, aggregate=np.sum)
  tempo_harm, beat_frames_harmonic = librosa.beat.beat_track(onset_envelope=harmonic_onset, sr=sr)
  beat_frames_harmnonic = librosa.core.frames_to_samples(beat_frames_harmonic)
  harmonic_onset_times = librosa.frames_to_time(np.arange(len(beat_frames_harmonic)), sr=sr)
'''
onset = librosa.onset.onset_strength(y, sr=sr, aggregate=np.mean)
tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr)
beat_frames = librosa.core.frames_to_samples(beats)
times = librosa.frames_to_time(np.arange(len(beats)), sr=sr)

cut(y, beat_frames)

data['sampleRate'] = sr
data['tempo'] = tempo 
data['onsets'] = times.tolist()
#data['perc_onsets'] = percussive_onset_times.tolist()
#data['harmonic_onsets'] = harmonic_onset_times.tolist()

json.dump(data, codecs.open(opts.dir + '/meta.json', 'w', encoding='utf-8'))

print(tempo)
#print(tempo_harm)
#print(tempo_perc)

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def pitchify(signal=y, begin=opts.begin, spread=opts.spread, int=opts.int): 
  for i in my_range(begin, spread, int):
    y_shift = librosa.effects.pitch_shift(singal, sr, i)
    librosa.output.write_wav(opts.dir + '/sample.' + str(i) + '.wav', y=y_shift, sr=sr)



