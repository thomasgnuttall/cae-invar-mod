run_name = 'icmpc_old_time'

import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
sns.set_theme()

from scipy.stats import spearmanr
from exploration.io import create_if_not_exists
from exploration.visualisation import flush_matplotlib

import csv
import json 
import os
import pickle
import yaml

#import essentia
#import essentia.standard
import numpy as np
import librosa
import soundfile as sf

from exploration.utils import get_timestamp

out_dir = f'/Volumes/Shruti/FOR_LARA/{run_name}/'
distances_gestures_path = os.path.join(out_dir, 'distances_gestures.csv')
all_groups_path = os.path.join(out_dir, 'all_groups.csv')
results_dir = os.path.join(out_dir, 'analysis', '')
results_df_path = os.path.join(results_dir, 'results.csv')

create_if_not_exists(results_df_path)

# load data
distances = pd.read_csv(distances_gestures_path)
all_groups = pd.read_csv(all_groups_path)

## Add features
###############
distances['1daccelerationDTWCombined'] = distances['1daccelerationDTWHand'] + distances['1daccelerationDTWForearm']
distances['2daccelerationDTWCombined'] = distances['2daccelerationDTWHand'] + distances['2daccelerationDTWForearm']
distances['1dvelocityDTWCombined'] = distances['1dvelocityDTWHand'] + distances['1dvelocityDTWForearm']
distances['2dvelocityDTWCombined'] = distances['2dvelocityDTWHand'] + distances['2dvelocityDTWForearm']


## Cluster
##########
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
def get_inter_intra(df, dcol):
    inter = df[df['cluster1']==df['cluster2']][dcol].mean()
    intra = df[df['cluster1']!=df['cluster2']][dcol].mean()
    return inter/intra

distance_metric = 'pitch_dtw'
t=50

just_dist = distances[['index1', 'index2', distance_metric]]
just_dist_flip = just_dist.copy()
just_dist_flip.columns = ['index2', 'index1', distance_metric]
just_dist = pd.concat([just_dist, just_dist_flip])

dist_piv = just_dist.pivot("index1", "index2", distance_metric).fillna(0)
indices = dist_piv.columns

piv_arr = dist_piv.values
X = piv_arr + np.transpose(piv_arr)

Z = linkage(squareform(X), 'ward')

metrics = []
all_ts = list(np.arange(10, 500))
for t in tqdm.tqdm(all_ts):
    clustering = fcluster(Z, t=t, criterion='distance', R=None, monocrit=None)

    clusterer = {x:y for x,y in zip(indices, clustering)}

    distances['cluster1'] = distances['index1'].apply(lambda y: clusterer[y])
    distances['cluster2'] = distances['index2'].apply(lambda y: clusterer[y])

    metrics.append(get_inter_intra(distances, distance_metric))

plot_path = os.path.join(out_dir, 'analysis', 'grouping', 'threshold_plot.png')
create_if_not_exists(plot_path)
plt.plot(all_ts, metrics)
plt.title('Varying t')
plt.xlabel('t')
plt.ylabel('Ratio of inter:intra cluster distance')
plt.savefig(plot_path)
flush_matplotlib()


final_t = 70
clustering = fcluster(Z, t=final_t, criterion='distance', R=None, monocrit=None)

clusterer = {x:y for x,y in zip(indices, clustering)}

distances['cluster1'] = distances['index1'].apply(lambda y: clusterer[y])
distances['cluster2'] = distances['index2'].apply(lambda y: clusterer[y])

print(f"Number of clusters: {max(clustering)}")
print(Counter(clustering))

## Get metrics
##############
distance_cols = ['pitch_dtw', 'diff_pitch_dtw', '1daccelerationDTWForearm', 
                 '2daccelerationDTWForearm', '1daccelerationDTWHand', 
                 '2daccelerationDTWHand', '1dvelocityDTWForearm', '2dvelocityDTWForearm', 
                 '1dvelocityDTWHand','2dvelocityDTWHand', '1daccelerationDTWCombined', 
                 '2daccelerationDTWCombined', '1dvelocityDTWCombined', '2dvelocityDTWCombined']
for col in distance_cols:
    met = round(get_inter_intra(distances, col), 3)
    print(f"{col}: {met}")


## Save new_distance matrix
###########################
frame_path = os.path.join(out_dir, 'distance_gesture_cluster.csv')
distances.to_csv(frame_path, index=False)


## Output audio and plots
#########################
cluster_out = os.path.join(out_dir, 'analysis', 'grouping', 'audios', '')

all_audios = {}
for t in all_groups['track'].unique():
    out_dir = os.path.join(f'/Volumes/Shruti/asplab2/cae-invar/data/self_similarity/{t}/')
    metadata_path = os.path.join(out_dir, 'metadata.pkl')
    metadata = load_pkl(metadata_path)
    audio_path = metadata['audio_path']
    all_audios[t],_ = librosa.load(audio_path)

sr = 44100
for i, row in all_groups.iterrows():
    index = row['index']
    start = row['start']
    end = row['end']
    group = row['group']
    occurrence = row['occurrence']
    track = row['track']
    tonic = row['tonic']

    cluster = clusterer[index]

    out_audio_path = os.path.join(cluster_out, f'group_{cluster}', f'start={round(start,2)}__end={round(end,2)}.wav')
    out_plot_path = os.path.join(cluster_out, f'group_{cluster}', f'start={round(start,2)}__end={round(end,2)}.png')

    create_if_not_exists(out_audio_path)

    ## write audio
    y = all_audios[track]
    s1 = start*sr
    s2 = end*sr
    subseq = y[int(s1):int(s2)]
    sf.write(out_audio_path, subseq, samplerate=sr)









import scipy
df = distances
for d in distance_cols:
    
    inter = df[df['cluster1']==df['cluster2']][d].values
    intra = df[df['cluster1']!=df['cluster2']][d].values
    res = scipy.stats.ttest_ind(inter, intra)
    print(f'{d}: statistic={round(res.statistic, 2)}, pvalue={round(res.pvalue, 2)}')







grouping_path = os.path.join(out_dir, 'analysis', 'grouping', 'groups', '')
import shutil
for i, row in all_groups.iterrows():
    index = row['index']
    start = row['start']
    end = row['end']
    group = row['group']
    occurrence = row['occurrence']
    track = row['track']
    tonic = row['tonic']
    cluster = row['cluster']
    

    if not np.isnan(cluster):
        this_dir = os.path.join(grouping_path, f'cluster_{int(cluster)}')
        create_if_not_exists(this_dir)
        folder = os.path.join(out_dir, track, 'results', run_name)

        all_f = [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder) for f in filenames]

        suffix = f'/{int(occurrence)}_'
    
        files = [x for x in all_f if suffix in x]

        for f in files:
            if not (('.wav' in f) or ('.png' in f)):
                continue
            filename = os.path.splitext(f)[0].split('/')[-1]
            filename = f'{track}___{filename}' + os.path.splitext(f)[1]
            new_filepath = os.path.join(this_dir, filename)
            create_if_not_exists(new_filepath)
            shutil.copyfile(f, new_filepath)


    

    files = []




all_groups['cluster'] = all_groups['index'].apply(lambda y: clusterer[y] if y in clusterer else np.nan)