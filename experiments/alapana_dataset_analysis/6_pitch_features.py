import pandas as pd
from exploration.pitch import get_timeseries, pitch_seq_to_cents,interpolate_below_length
from exploration.io import create_if_not_exists
import pandas as pd
import numpy
import librosa
import os
import fastdtw
import numpy as np
import tqdm
import dtaidistance.dtw
import soundfile as sf
from experiments.alapana_dataset_analysis.dtw import dtw_path, dtw_dtai
from scipy.ndimage import gaussian_filter1d

import librosa

run_name = 'result_0.1'

### SPECTRAL FLUX - MFCC
### SPECTRAL FLUX - MFCC
### SPECTRAL FLUX - MFCC
### SPECTRAL FLUX - MFCC
### SPECTRAL FLUX - MFCC
### SPECTRAL FLUX - MFCC

out_dir = f'/Volumes/MyPassport/FOR_LARA/{run_name}/'

metadata_path = 'audio/lara_wim2/info.csv'
metadata = pd.read_csv(metadata_path)
metadata = metadata[~metadata['Audio file'].isnull()]
metadata = metadata[~metadata['Tonic'].isnull()]
tonic_dict = {k:v for k,v in metadata[['Audio file', 'Tonic']].values}

all_groups = pd.read_csv(os.path.join(out_dir, 'all_groups.csv'))

def get_tonic(t, metadata):
    tonic = metadata[metadata['Audio file'].apply(lambda x: x in t)]['Tonic'].values[0]
    return tonic

# get audios and pitches
audio_tracks = {}
pitch_tracks = {}
for t in all_groups['track'].unique():
    a_path = f'/Volumes/MyPassport/cae-invar/audio/lara_wim2/spleeter/{t}.mp3'
    audio_tracks[t], _ = librosa.load(a_path, sr=44100)
    
    p_path = f'/Volumes/MyPassport/cae-invar/data/pitch_tracks/alapana/{t}.csv'
    tonic = get_tonic(t, metadata)
    pitch, time, timestep = get_timeseries(p_path)
    pitch = pitch_seq_to_cents(pitch, tonic)
    pitch[pitch==None]=0
    pitch = interpolate_below_length(pitch, 0, (350*0.001/timestep))
    pitch = pitch.astype(float)
    pitch_tracks[t] = (gaussian_filter1d(pitch, 2.5), time, timestep)



def get_loudness(y, window_size=2048):
    S = librosa.stft(y, n_fft=window_size)**2
    power = np.abs(S)**2
    p_mean = np.sum(power, axis=0, keepdims=True)
    p_ref = np.max(power)
    loudness = librosa.power_to_db(p_mean, ref=p_ref)
    return loudness[0]


# distance from tonic tracks
dtonic_tracks = {}
# spectral flux tracks
sflux_tracks = {}
# loudness tracks
loudness_tracks = {}
for t in all_groups['track'].unique():
    y = audio_tracks[t]
    pitch, time, timestep = pitch_tracks[t]

    # loudness
    loudness = get_loudness(y)
    step = len(y)/len(loudness)
    loudness_smooth = gaussian_filter1d(loudness, 10)
    loudness_smooth -= loudness_smooth.min()
    loudness_tracks[t] = (loudness_smooth, step)

    # Change in loudness

    # Distance from tonic
    sameoctave = pitch%1200
    upper = 1200-sameoctave
    stack = np.vstack((upper, sameoctave)).T
    dtonic = stack.min(axis=1)
    if True in np.isnan(dtonic):
        break
    dtonic_tracks[t] = (dtonic, time, timestep)



# window_size = 2048
# loudness = get_loudness(y, window_size)
# step = len(y)/len(loudness)
# loudness_smooth = gaussian_filter1d(loudness, 10)
# loudness_smooth -= loudness_smooth.min()

# s1=176
# s2=186
# sr=44100
# ltrack = loudness_smooth[int(s1*sr/step):int(s2*sr/step)]
# atrack = y[int(s1*sr):int(s2*sr)]
# import matplotlib.pyplot as plt

# plt.plot(list(range(len(ltrack))), ltrack)
# plt.ylabel('Normalised Loudness (dB)')
# plt.xlabel('Time')
# plt.savefig('loudness_smooth.png')
# plt.clf()
# # Write out audio as 24bit PCM WAV
# sf.write('loudness_smooth.wav', atrack, sr, subtype='PCM_24')



sr=44100

audio_distances_path = os.path.join(out_dir, 'audio_distances.csv')

try:
    print('Removing previous distances file')
    os.remove(audio_distances_path)
except OSError:
    pass
create_if_not_exists(audio_distances_path)

##text=List of strings to be written to file
header = 'index1,index2,loudness_dtw,distance_from_tonic_dtw'
with open(audio_distances_path,'a') as file:
    file.write(header)
    file.write('\n')

    for i, row in tqdm.tqdm(list(all_groups.iterrows())):

        qstart = row.start
        qend = row.end
        qtrack = row.track
        qindex = row['index']

        (qloudness, qloudnessstep) = loudness_tracks[qtrack]
        (qdtonic, qtime, qtimestep) = dtonic_tracks[qtrack]

        loudness_sq1 = int(qstart*sr/qloudnessstep)
        loudness_sq2 = int(qend*sr/qloudnessstep)
        dtonic_sq1 = int(qstart/qtimestep)
        dtonic_sq2 = int(qend/qtimestep)
        for j, rrow in all_groups.iterrows():
                rstart = rrow.start
                rend = rrow.end
                rtrack = rrow.track
                rindex = rrow['index']
                if qindex <= rindex:
                    
                    continue
                (rloudness, rloudnessstep) = loudness_tracks[rtrack]
                (rdtonic, rtime, rtimestep) = dtonic_tracks[rtrack]

                loudness_sr1 = int(rstart*sr/rloudnessstep)
                loudness_sr2 = int(rend*sr/rloudnessstep)
                dtonic_sr1 = int(rstart/rtimestep)
                dtonic_sr2 = int(rend/rtimestep)

                pat1_loudness = qloudness[loudness_sq1:loudness_sq2]
                pat2_loudness = rloudness[loudness_sr1:loudness_sr2]
                
                pat1_dtonic = qdtonic[dtonic_sq1:dtonic_sq2]
                pat2_dtonic = rdtonic[dtonic_sr1:dtonic_sr2]

                # DTW normal loudness
                p1l = len(pat1_loudness)
                p2l = len(pat2_loudness)

                l_longest = max([p1l, p2l])
                l_shortest = min([p1l, p2l])
                #if l_longest/l_shortest-1 > 0.5:
                #    continue
                path, dtw_val = path, dtw_val = dtw_dtai(pat1_loudness, pat2_loudness, r=0.1)
                l = len(path)
                loudness_dtw = dtw_val/l
                
                # DTW normal dtonic
                p1l = len(pat1_dtonic)
                p2l = len(pat2_dtonic)

                l_longest = max([p1l, p2l])
                path, dtw_val = path, dtw_val = dtw_dtai(pat1_dtonic, pat2_dtonic, r=0.1)
                l = len(path)
                dtonic_dtw = dtw_val/l

                # Write
                line =f"{qindex},{rindex},{loudness_dtw},{dtonic_dtw}"

                file.write(line)
                file.write('\n')



