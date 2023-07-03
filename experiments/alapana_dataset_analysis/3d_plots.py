import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy
from numpy.random import randn
from scipy import array, newaxis

out_dir = f'/Volumes/MyPassport/FOR_LARA/{run_name}/'
all_feature_path2 = os.path.join(out_dir, 'all_features.csv')

i1 = 1119
i2 = 640

hand1 = all_groups[all_groups['index']==i1].iloc[0]['handedness']
hand2 = all_groups[all_groups['index']==i2].iloc[0]['handedness']
pelvis1 = all_groups[all_groups['index']==i1].iloc[0]['pelvis_centroid']
pelvis2 = all_groups[all_groups['index']==i2].iloc[0]['pelvis_centroid']

v1 = get_motion_data(i1, 'Position', 'Hand', hand1, 1, pelvis1)
v2 = get_motion_data(i2, 'Position', 'Hand', hand2, 1, pelvis2)


pi = len(v1)
pj = len(v2) 
l_longest = max([pi, pj])
#dtw_val, path = dtaidistance.dtw.distance(v1, v2, window=round(l_longest*0.20), use_c=True)
#dtw_val, path = fastdtw.fastdtw(v1, v2, dist=None, radius=round(l_longest*0.20))
path, dtw_val = dtw_path(v1, v2, radius=round(l_longest*0.1))

computed_val = dtw_val/len(path)

audio_distance_cut = pd.read_csv(all_feature_path2)
df_val = audio_distance_cut[(audio_distance_cut['index1']==i1) & (audio_distance_cut['index2']==i2)].iloc[0]['1dpositionDTWHand']

print(f'computed: {computed_val}')
print(f'df: {df_val}')


# =====
## data
fig, axs = plt.subplots(2)
fig.tight_layout()
axs[0].plot(v1)
axs[1].plot(v2)

axs[0].set_title(f'index: {i1}, 1dpositionDTWHand')
axs[1].set_title(f'index: {i2}, 1dpositionDTWHand')

plt.savefig('stacked.png')
plt.close('all')