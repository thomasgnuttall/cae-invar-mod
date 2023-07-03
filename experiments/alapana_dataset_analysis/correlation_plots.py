run_name = 'result_0.1'

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

out_dir = f'/Volumes/MyPassport/FOR_LARA/{run_name}/'

all_groups_path = os.path.join(out_dir, 'all_groups.csv')
sig_results_df_path = os.path.join(results_dir, 'results_significant.csv')
all_feature_path = os.path.join(results_dir, 'all_features.csv')

# load data
results = pd.read_csv(sig_results_df_path)
all_groups = pd.read_csv(all_groups_path)
all_features = pd.read_csv(all_feature_path)

targets = ['pitch_dtw', 'diff_pitch_dtw', 'loudness_dtw']

features = [
       '1dpositionDTWHand',
       '3dpositionDTWHand', '1dvelocityDTWHand',
       '3dvelocityDTWHand', '1daccelerationDTWHand',
       '3daccelerationDTWHand']



# for each target
for t in targets:
	for level in ['performer', 'performance', 'all']:
		df = results[
			(results['level']==level) &
			(results['x']==t)
		][['y', 'level_value', 'corr']]
		
		df = df.set_index(['y', 'level_value'])['corr']

		ax = df.unstack().plot(kind='bar')
		
		fig = ax.get_figure()
		fig.set_size_inches(15,7)
		fig.subplots_adjust(bottom=0.4, right=1)
		plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees

		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 10})
		text = f"on a {level} level" if not level == 'all' else 'for all performers/performances'
		ax.set_title(f'Signficant correlations between feature and {t} {text}')
		ax.set_xlabel(f'Feature')
		ax.set_ylabel(f'SRCC')
		ax.figure.savefig(f'histograms/correlation_plot__target={t}_level={level}.png',bbox_inches='tight')


# bar plots for each feature

# performace level

# performer level