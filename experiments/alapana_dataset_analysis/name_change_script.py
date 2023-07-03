# load laras mapping

# load all groups

# for each index
    # get current filepath of audio and plot
    # create new filepath
        # index, track, name, start, end
    # change in all_groups
    # store in new location

import pandas as pd
import os
import shutil

import tqdm

from exploration.io import create_if_not_exists

run_name = 'icmpc_old_time'

new_folder_name = 'results'
new_results_folder = f'../FOR_LARA/{new_folder_name}'

mapping = pd.read_csv('../FOR_LARA/info_name_mapping_LP.csv')
all_groups = pd.read_csv(f'../FOR_LARA/{run_name}/all_groups.csv')

bad_indices = []
for i,row in tqdm.tqdm(list(all_groups.iterrows())):
    index = row['index']
    start = row.start
    end = row.end
    group = row.group
    occurrence = row.occurrence
    track = row.track
    tonic = row.tonic
    
    new_track_name = mapping[mapping['Audio file']==track]['display_name'].values[0]

    folder = f'../FOR_LARA/{run_name}/{track}/results/{run_name}/'

    files = list(os.walk(folder))

    motifs_folder = [(dirpath, dirnames, filenames) for (dirpath, dirnames, filenames) in files if 'motif_0' in dirpath]
    all_filenames = motifs_folder[0][2]

    prefix = f'{int(occurrence)}_'
    lp = len(prefix)

    filenames = [f for f in all_filenames if f[:lp]==prefix if f[-4:] in ['.png', '.wav']]

    if len(filenames) == 0:
        bad_indices.append(index)
        continue

    for f in filenames:
        if '.png' in f:
            suffix = '.png'
        if '.wav' in f:
            suffix = '.wav'
        old_path = f'{motifs_folder[0][0]}/{f}'
        new_path = f'{new_results_folder}/plots_and_audio/index={index}__performance={new_track_name}__start={round(start)}__end={round(end)}{suffix}'
        create_if_not_exists(new_path)
        shutil.copyfile(old_path, new_path)

all_groups = all_groups[~all_groups['index'].isin(bad_indices)]
all_groups['display_name'] = all_groups['track'].apply(lambda y: mapping[mapping['Audio file']==y]['display_name'].values[0])
all_groups = all_groups[['index', 'start', 'end', 'track', 'display_name', 'tonic']]


all_groups = pd.read_csv(f'../FOR_LARA/{run_name}/all_groups.csv')
all_groups.to_csv(os.path.join(new_results_folder, 'all_groups.csv'), index=False)

distances = pd.read_csv(f'../FOR_LARA/{run_name}/distances_with_performers.csv')
distances = distances[['index1', 'index2', 'performance1', 'performer1', 'length1', 'performance2', 
        'performer2', 'length2', 'length_diff', 'pitch_dtw', 'diff_pitch_dtw',
       '1daccelerationDTWForearm', '2daccelerationDTWForearm',
       '1daccelerationDTWHand', '2daccelerationDTWHand',
       '1dvelocityDTWForearm', '2dvelocityDTWForearm', '1dvelocityDTWHand',
       '2dvelocityDTWHand', 'pitch_dtw_log',
       'diff_pitch_dtw_log', '1daccelerationDTWForearm_log',
       '2daccelerationDTWForearm_log', '1daccelerationDTWHand_log',
       '2daccelerationDTWHand_log', '1dvelocityDTWForearm_log',
       '2dvelocityDTWForearm_log', '1dvelocityDTWHand_log',
       '2dvelocityDTWHand_log']]

distances['performance1'] = distances['performance1'].apply(lambda y: mapping[mapping['Audio file']==y]['display_name'].values[0])
distances['performance2'] = distances['performance2'].apply(lambda y: mapping[mapping['Audio file']==y]['display_name'].values[0])

distances['performer1'] = distances['performance1'].apply(lambda y: y.split('_')[0])
distances['performer2'] = distances['performance2'].apply(lambda y: y.split('_')[0])

distances.to_csv(os.path.join(new_results_folder, 'distances.csv'), index=False)



