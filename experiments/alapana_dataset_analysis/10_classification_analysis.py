run_name = 'results'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
sns.set_theme()

from scipy.stats import spearmanr
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

levels = ['all']

all_features['same_pattern'] = all_features['pitch_dtw']<90

results = pd.DataFrame(columns=['target', 'same_thresh', 'level', 'level_value', 'test_score', 'train_score', 'best_params', 'n_samples'])

target = 'same_pattern'
for level in levels:
    print(level)
    cols = [x for x in all_features.columns if level in x]
    uniqs = set()
    for c in cols:
        uniqs = uniqs.union(set(all_features[c].unique()))
    if level == 'all':
        uniqs = [0]
    for u in uniqs:
        if level == 'all':
            this_df = all_features
        else:
            this_df = all_features[(all_features[cols[0]]==u) & (all_features[cols[1]]==u)]
        if len(this_df) > 200:
              X = this_df[features].values
              y = this_df[target].values

              X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

              # weight samples for unbalanced set
              sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

              # Predictor has many parameters, we want to select the best so we train 
              # a model for each combination of the following parameters
              # selecting the best as it is evaluated on our training set
              all_params = {
                  "n_estimators": [50, 100, 150, 200],
                  'learning_rate': [0.001, 0.01, 0.1],
                  'max_depth': [8, 10, 12, 15],
                  'min_samples_leaf': [5, 10, 15, 20]
              }

              # Initialise predictor
              gbc = GradientBoostingClassifier()

              # set up gridsearch (will train one classifier for each combination of parameters)
              clf = GridSearchCV(gbc, all_params, cv=3, scoring='f1')

              # fit for just good features (might take some time)
              clf.fit(X_train, y_train)
              test_score = clf.score(X_test, y_test)
              train_score = clf.best_score_

              to_append = {
                  'target': target,
                  'thresh':same_thresh,
                  'level': level,
                  'level_value': u,
                  'test_score': test_score,
                  'train_score': train_score,
                  'best_params': clf.best_params_,
                  'n_samples': len(this_df),
                  'train_imbalance': sum(y_train)/len(y_train),
                  'test_imbalance': sum(y_test)/len(y_test)
              }
              results = results.append(to_append, ignore_index=True)

results.to_csv('/Volumes/MyPassport/cae-invar/experiments/alapana_dataset_analysis/results/results_f1_first.csv', index=False)