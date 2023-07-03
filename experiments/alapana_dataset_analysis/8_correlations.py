run_name = 'result_0.1'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
sns.set_theme()

from scipy.stats import spearmanr
from exploration.io import create_if_not_exists, load_pkl

out_dir = f'/Volumes/MyPassport/FOR_LARA/{run_name}/'
distances_path = os.path.join(out_dir, 'distances_gestures.csv')
index_features_path = os.path.join(out_dir, 'index_features.pkl')
all_groups_path = os.path.join(out_dir, 'all_groups.csv')
audio_features_path = os.path.join(out_dir, 'audio_distances.csv')
results_dir = os.path.join(out_dir, 'analysis', '')
results_df_path = os.path.join(results_dir, 'results.csv')
sig_results_df_path = os.path.join(results_dir, 'results_significant.csv')
all_feature_path = os.path.join(results_dir, 'all_features.csv')
all_feature_path2 = os.path.join(out_dir, 'all_features.csv')

create_if_not_exists(results_df_path)

# load data
distances = pd.read_csv(distances_path)
all_groups = pd.read_csv(all_groups_path)
audio_features = pd.read_csv(audio_features_path)
index_features = load_pkl(index_features_path)

pitch_targets = ['pitch_dtw', 'diff_pitch_dtw']

audio_targets = ['loudness_dtw']

features = [
       '1dpositionDTWHand',
       '3dpositionDTWHand', '1dvelocityDTWHand',
       '3dvelocityDTWHand', '1daccelerationDTWHand',
       '3daccelerationDTWHand']

audio_distance = distances.merge(audio_features, on=['index1', 'index2'])

targets = pitch_targets + audio_targets

## Remove mismatched length for analysis
########################################
def len_mismatch(l1, l2):
    l_longest = max([l1, l2])
    l_shortest = min([l1, l2])

    return l_longest/l_shortest-1 > 0.5

audio_distance['length_mismatch'] = audio_distance.apply(lambda y: len_mismatch(y.length1, y.length2), axis=1)

audio_distance_cut = audio_distance[audio_distance['length_mismatch']!=True]

audio_distance.to_csv(all_feature_path, index=False)
audio_distance.to_csv(all_feature_path2, index=False)

#audio_distance_cut = audio_distance

## Get correlations
###################
def get_correlation(df, x, y, level=None):
    corr_dict = {}
    if level:
        cols = [x for x in df.columns if level in x]
        uniqs = set()
        for c in cols:
            uniqs = uniqs.union(set(df[c].unique())) 
        for u in uniqs:
            this_df = df[(df[cols[0]]==u) & (df[cols[1]]==u)]
            res = spearmanr(a=this_df[x].values, b=this_df[y].values, axis=0, nan_policy='omit')
            corr_dict[u] = {'corr': res.correlation, 'p': res.pvalue, 'n': len(this_df)} 
    else:
        res = spearmanr(a=df[x].values, b=df[y].values, axis=0, nan_policy='omit')
        corr_dict['all'] = {'corr': res.correlation, 'p': res.pvalue, 'n': len(df)}
    return corr_dict


levels = [None, 'performer', 'performance']

results = pd.DataFrame(columns=['x', 'y', 'level', 'level_value', 'corr', 'p', 'n'])
for x in targets:
    print(x)
    for y in tqdm.tqdm(features):
        for level in levels:
            corr_dict = get_correlation(audio_distance_cut, x, y, level)
            for k,v in corr_dict.items():
                to_append = {
                    'x': x,
                    'y': y,
                    'level': level if level else 'all',
                    'level_value': k,
                    'corr': v['corr'],
                    'p': v['p'],
                    'n': v['n']
                }
                results = results.append(to_append, ignore_index=True)

results = results.sort_values(by='corr', ascending=False)

results.to_csv(results_df_path, index=False)

## Best Results
###############
# For 70% of estimates to be within +/- 0.1 of the true correlation value (between -0.1 and 0.1), we need at least 109 observations
# for 90% of estimates to be within +/- 0.2 of the true correlation value (between -0.2 and 0.2), we need at least 70 observations. 
p = 0.01
significant_results = results[(results['p']<p) & (results['n']>109)]
significant_results.to_csv(sig_results_df_path, index=False)

## Plots
########
for i, row in tqdm.tqdm(list(significant_results.iterrows())):
    x = row['x']
    y = row['y']
    level = row['level']
    level_value = row['level_value']
    corr = row['corr']
    p = row['p']
    n = row['n']

    level_str = f'{level}={level_value}' if level != 'all' else level
    title = f'{x} against {y} for {level_str}'
    title += f'\n [spearmans R={round(corr,3)}, p={round(p,3)}, n={n}]'
    
    out_path = os.path.join(results_dir, 'plots', x, y, level, (level if level == 'all' else str(level_value)) + '.png')
    create_if_not_exists(out_path)

    if level != 'all':
        cols = [x for x in audio_distance_cut.columns if level in x]
        data = audio_distance_cut[(audio_distance_cut[cols[0]]==level_value) & (audio_distance_cut[cols[1]]==level_value)]
    else:
        data = audio_distance_cut

    pl = sns.scatterplot(data, x=x, y=y, s=5)
    pl.set_title(title)
    fig = pl.get_figure()
    fig.savefig(out_path)
    plt.cla()
    plt.clf()
    plt.close()

