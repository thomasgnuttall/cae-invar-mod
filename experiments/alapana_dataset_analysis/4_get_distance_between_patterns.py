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

from experiments.alapana_dataset_analysis.dtw import dtw_path
from scipy.ndimage import gaussian_filter1d

run_name = 'icmpc_old_time'


out_dir = f'/Volumes/Shruti/FOR_LARA/{run_name}/'


metadata_path = 'audio/lara_wim2/info.csv'
metadata = pd.read_csv(metadata_path)
metadata = metadata[~metadata['Audio file'].isnull()]
metadata = metadata[~metadata['Tonic'].isnull()]
tonic_dict = {k:v for k,v in metadata[['Audio file', 'Tonic']].values}

def get_tonic(t, metadata):
    tonic = metadata[metadata['Audio file'].apply(lambda x: x in t)]['Tonic'].values[0]
    return tonic

def get_raga(t, metadata):
    raga = metadata[metadata['Audio file'].apply(lambda x: x in t)]['Raga'].values[0]
    return raga

def get_derivative(pitch, time):

    d_pitch = np.diff(pitch) / np.diff(time)
    d_time = (np.array(time)[:-1] + np.array(time)[1:]) / 2

    return d_pitch, d_time

all_patts = pd.DataFrame()
pitch_tracks = {}
for t in [x for x in track_names if x not in failed_tracks]:
    if not "2018_11_19" in t:
        g_path = os.path.join(out_dir,t,"results",run_name,"groups.csv")
        p_path = f'/Volumes/Shruti/asplab2/cae-invar/data/pitch_tracks/alapana/{t}.csv'
        patts = pd.read_csv(g_path)
        patts['track'] = t
        tonic = get_tonic(t, metadata)
        patts['tonic'] = tonic
        all_patts = pd.concat([all_patts, patts])
        pitch, time, timestep = get_timeseries(p_path)
        pitch = pitch_seq_to_cents(pitch, tonic=tonic)
        pitch[pitch==None]=0
        pitch = interpolate_below_length(pitch, 0, (350*0.001/timestep))
        pitch_d, time_d = get_derivative(pitch, time)
        pitch_tracks[t] = (gaussian_filter1d(pitch, 2.5), time, timestep, gaussian_filter1d(pitch_d, 2.5), time_d)


all_patts.reset_index(inplace=True)
all_patts['index'] = range(0,len(all_patts))

min_length = 2
bad_indices = []
for i, row in all_patts.iterrows():
    if row.end - row.start < min_length:
        bad_indices.append(row['index'])


all_patts.to_csv(os.path.join(out_dir,'all_groups.csv'), index=False)

#all_distances = pd.DataFrame(columns=['index1', 'index2', 'path1_start', 'path1_end', 'path2_start', 'path2_end', 'path_length', 'dtw_distance', 'dtw_distance_norm'])


distances_path = os.path.join(out_dir, 'distances.csv')

try:
    print('Removing previous distances file')
    os.remove(distances_path)
except OSError:
    pass
create_if_not_exists(distances_path)

##text=List of strings to be written to file
header = 'index1,index2,pitch_dtw,diff_pitch_dtw,pitch_dtw_dtai,diff_pitch_dtw_dtai'
with open(distances_path,'a') as file:
    file.write(header)
    file.write('\n')

    for i, row in tqdm.tqdm(list(all_patts.iterrows())):

        qstart = row.start
        qend = row.end
        qtrack = row.track

        (qpitch, qtime, qtimestep, qpitch_d, qtime_d) = pitch_tracks[qtrack]

        sq1 = int(qstart/qtimestep)
        sq2 = int(qend/qtimestep)
        for j, rrow in all_patts.iterrows():
            if (i in bad_indices) or (j in bad_indices):
                continue
            if i < j:
                rstart = rrow.start
                rend = rrow.end
                rtrack = rrow.track

                (rpitch, rtime, rtimestep, rpitch_d, rtime_d) = pitch_tracks[rtrack]
                sr1 = int(rstart/rtimestep)
                sr2 = int(rend/rtimestep)

                pat1 = qpitch[sq1:sq2]
                pat2 = rpitch[sr1:sr2]
                
                pat1[pat1 == None] = 0
                pat2[pat2 == None] = 0

                pat1 = np.trim_zeros(pat1)
                pat2 = np.trim_zeros(pat2)

                diff1 = qpitch_d[sq1:sq2]
                diff2 = rpitch_d[sr1:sr2]

                # DTW normal
                p1l = len(pat1) 
                p2l = len(pat2)

                l_longest = max([p1l, p2l])
                dtw_val, path = fastdtw.fastdtw(pat1, pat2, dist=None, radius=round(l_longest*0.20))
                l = len(path)
                dtw_norm = dtw_val/l

                # DTW diff
                p1l = len(diff1) 
                p2l = len(diff2)

                l_longest = max([p1l, p2l])
                dtw_val, path = fastdtw.fastdtw(diff1, diff2, dist=None, radius=round(l_longest*0.20))
                l = len(path)
                dtw_norm_diff = dtw_val/l

                # DTW normal DTAI
                p1l = len(pat1) 
                p2l = len(pat2)

                l_longest = max([p1l, p2l])
                dtw_norm_dtai = dtaidistance.dtw.distance(pat1, pat2, window=round(l_longest*0.20), use_c=True, psi=round(l_longest*0.2))

                # DTW diff DTAI
                p1l = len(diff1) 
                p2l = len(diff2)

                l_longest = max([p1l, p2l])
                dtw_norm_diff_dtai = dtaidistance.dtw.distance(diff1, diff2, window=round(l_longest*0.20), use_c=True, psi=round(l_longest*0.2))

                line =f"{i},{j},{dtw_norm},{dtw_norm_diff},{dtw_norm_dtai},{dtw_norm_diff_dtai}"
                #all_distances = all_distances.append({
                #   'index1':i,
                #   'index2':j,
                #   'path1_start':path1_start,
                #   'path1_end':path1_end,
                #   'path2_start':path2_start,
                #   'path2_end':path2_end,
                #   'path_length': l,
                #   'dtw_distance':dtw_val,
                #   'dtw_distance_norm':dtw_norm
                #}, ignore_index=True)
                file.write(line)
                file.write('\n')

    #all_distances.reset_index(inplace=True, drop=True)



