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
    cmd = f'lame --preset insane {path}'
    subprocess.call(cmd, shell=True)

    # delete original
    cmd = f'rm {path}'
    subprocess.call(cmd, shell=True)

def main(p):
    try:
        #dataset_dir = '/Volumes/Shruti/data/compmusic/Carnatic/audio/'
        vocal_dataset_dir = '/Volumes/Shruti/data/compmusic/Carnatic_vocal/audio/'


        #paths = get_all_paths(dataset_dir)

        audio_path = p#os.path.join(dataset_dir, p)
        out_path = os.path.join(vocal_dataset_dir, p).replace('mp3', 'wav')
        create_if_not_exists(out_path)

        vocal = isolate_vocal(audio_path, sr)
        
        sf.write(out_path, vocal, samplerate=sr)

        convert_wav_mp3(out_path)
        print(f'{p} complete')
    except Exception as e:
        print(f'{p} failed with error {e}')

if __name__ == '__main__':
    p = sys.argv[1]
    print(p)
    main(p)