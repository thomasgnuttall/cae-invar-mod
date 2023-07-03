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

out_dir = f'/Volumes/MyPassport/FOR_LARA/{run_name}/'

results_dir = os.path.join(out_dir, 'analysis', '')
results_df_path = os.path.join(results_dir, 'results.csv')
sig_results_df_path = os.path.join(results_dir, 'results_significant.csv')

create_if_not_exists(results_df_path)

# load data
results = pd.read_csv(sig_results_df_path)

features = [
      '1dpositionDTWHand',
      '3dpositionDTWHand',
      '1dvelocityDTWHand',
      '3dvelocityDTWHand',
      '1daccelerationDTWHand', 
      '3daccelerationDTWHand']

results = results[results['y'].isin(features)]

