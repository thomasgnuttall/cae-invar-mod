import os
import sys

import numpy as np
import pandas as pd
import tqdm

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

import essentia.standard as estd

import soundfile as sf

import subprocess

from pathlib import Path

sr = 44100

def create_if_not_exists(path):
    """
    If the directory at <path> does not exist, create it empty
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ''):
        os.makedirs(directory)


def isolate_vocal(audio_path, sr=sr):
    # Run spleeter on track to remove the background
    separator = Separator('spleeter:2stems')
    audio_loader = AudioAdapter.default()

    waveform, _ = audio_loader.load(audio_path, sample_rate=sr)
    prediction = separator.separate(waveform=waveform)
    clean_vocal = prediction['vocals']

    # Processing
    audio_mono = clean_vocal.sum(axis=1) / 2
    audio_mono_eqloud = estd.EqualLoudness(sampleRate=sr)(audio_mono)

    return audio_mono_eqloud


def get_all_paths(d):
    p = Path(d)
    paths = [str(x).replace(d,'') for x in p.glob('**/*') if '.mp3' in str(x)]
    return paths


def convert_wav_mp3(path):
    # convert
    cmd = f'lame --preset insane "{path}"'
    subprocess.call(cmd, shell=True)

    # delete original
    cmd = f'rm "{path}"'
    subprocess.call(cmd, shell=True)



carnatic = os.listdir('/Volumes/Shruti/data/compmusic/Carnatic/audio/')

carnatic_vocal = os.listdir('/Volumes/Shruti/data/compmusic/Carnatic_vocal/audio/')

missing = [x for x in carnatic if x not in carnatic_vocal]

old_dir = '/Volumes/Shruti/data/compmusic/Carnatic/audio/'
new_dir = '/Volumes/Shruti/data/compmusic/Carnatic_vocal_missing/audio/'

p = Path(old_dir)
all_paths = [str(x).replace(old_dir, '') for x in p.glob('**/*') if '.mp3' in str(x)]
missing_paths = [p for p in all_paths if any([l in p for l in missing])]

sr = 44100

with open('experiments/missing_carnatic.sh', 'a') as f:
    for mp in missing_paths:
        in_path = os.path.join(old_dir, mp)
        out_path = os.path.join(new_dir, mp)
        f.write(f'python experiments/spleet.py "{in_path}" "{out_path}"\n')

