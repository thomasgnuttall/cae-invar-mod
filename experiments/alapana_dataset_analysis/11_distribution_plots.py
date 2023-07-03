run_name = 'results'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
sns.set_theme()

from scipy.stats import kstest
from exploration.io import create_if_not_exists
from exploration.visualisation import flush_matplotlib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.utils.class_weight import compute_sample_weight

out_dir = f'/Volumes/MyPassport/FOR_LARA/{run_name}/'
distances_path = os.path.join(out_dir, 'distances.csv')
all_groups_path = os.path.join(out_dir, 'all_groups.csv')
audio_distances_path = os.path.join(out_dir, 'audio_distances.csv')

distances = pd.read_csv(distances_path)
all_groups = pd.read_csv(all_groups_path)
audio_distances = pd.read_csv(audio_distances_path)

all_features = distances.merge(audio_distances, on=['index1','index2'])

all_features['same_pattern'] = all_features['pitch_dtw']<90

features = [
       '1daccelerationDTWForearm',
       '2daccelerationDTWForearm', '1daccelerationDTWHand',
       '2daccelerationDTWHand', '1dvelocityDTWForearm', '2dvelocityDTWForearm',
       '1dvelocityDTWHand', '2dvelocityDTWHand'
       ]

targets = [
    'pitch_dtw',
    'diff_pitch_dtw', 
    'loudness_dtw', 
    'distance_from_tonic_dtw'
]

# get distributions
feature = '2dvelocityDTWHand'
same_pop = all_features[all_features['same_pattern']==True][feature].values
diff_pop = all_features[all_features['same_pattern']==False][feature].values

plt.hist(np.log(same_pop), bins=100, alpha=0.3, density=True)
plt.hist(np.log(diff_pop), bins=100, alpha=0.3, density=True)
plt.show()
plt.clf()