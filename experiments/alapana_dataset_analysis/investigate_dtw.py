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

from matplotlib.ticker import NullFormatter, FormatStrFormatter
import matplotlib.pyplot as plt

from experiments.alapana_dataset_analysis.dtw import dtw_path
from scipy.ndimage import gaussian_filter1d

run_name = 'results'


out_dir = f'/Volumes/MyPassport/FOR_LARA/{run_name}/'

track_names = [
    "2018_11_13_am_Sec_1_P2_Varaali_slates",
    "2018_11_13_am_Sec_2_P1_Shankara_slates",
    "2018_11_13_am_Sec_3_P1_Anandabhairavi_slates",
    "2018_11_13_am_Sec_3_P3_Kalyani_slates",
    "2018_11_13_am_Sec_3_P5_Todi_slates",
    "2018_11_13_am_Sec_3_P8_Bilahari_B_slates",
    "2018_11_13_am_Sec_4_P1_Atana_slatesB",
    "2018_11_13_pm_Sec_1_P1_Varaali_slates",
    "2018_11_13_pm_Sec_1_P2_Anandabhairavi_slates",
    "2018_11_13_pm_Sec_2_P1_Kalyani_slates",
    "2018_11_13_pm_Sec_3_P1_Todi_A_slates",
    "2018_11_13_pm_Sec_3_P2_Todi_B_slates",
    "2018_11_13_pm_Sec_4_P1_Shankara_slates",
    "2018_11_13_pm_Sec_5_P1_Bilahari_slates",
    "2018_11_13_pm_Sec_5_P2_Atana_slates",
    "2018_11_15_Sec_10_P1_Atana_slates",
    "2018_11_15_Sec_12_P1_Kalyani_slates",
    "2018_11_15_Sec_13_P1_Bhairavi_slates",
    "2018_11_15_Sec_1_P1_Anandabhairavi_A_slates",
    "2018_11_15_Sec_2_P1_Anandabhairavi_B_slates",
    "2018_11_15_Sec_3_P1_Anandabhairavi_C_slates",
    "2018_11_15_Sec_4_P1_Shankara_slates",
    "2018_11_15_Sec_6_P1_Varaali_slates",
    "2018_11_15_Sec_8_P1_Bilahari_slates",
    "2018_11_15_Sec_9_P1_Todi_slates",
    "2018_11_18_am_Sec_1_P1_Varaali_slates",
    "2018_11_18_am_Sec_2_P2_Shankara_slates",
    "2018_11_18_am_Sec_3_P1_Anandabhairavi_A_slates",
    "2018_11_18_am_Sec_3_P2_Anandabhairavi_B_slates",
    "2018_11_18_am_Sec_4_P1_Kalyani_slates",
    "2018_11_18_am_Sec_5_P1_Bhairavi_slates",
    "2018_11_18_am_Sec_5_P3_Bilahari_slates",
    "2018_11_18_am_Sec_6_P1_Atana_A_slates",
    "2018_11_18_am_Sec_6_P2_Atana_B_slates",
    "2018_11_18_am_Sec_6_P4_Todi_slates",
    "2018_11_18_am_Sec_6_P6_Bilahari_B",
    "2018_11_18_pm_Sec_1_P1_Shankara_slates",
    "2018_11_18_pm_Sec_1_P2_Varaali_slates",
    "2018_11_18_pm_Sec_1_P3_Bilahari_slates",
    "2018_11_18_pm_Sec_2_P2_Anandabhairavi_full_slates",
    "2018_11_18_pm_Sec_3_P1_Kalyani_slates",
    "2018_11_18_pm_Sec_4_P1_Bhairavi_slates",
    "2018_11_18_pm_Sec_4_P2_Atana_slates",
    "2018_11_18_pm_Sec_5_P1_Todi_full_slates",
    "2018_11_18_pm_Sec_5_P2_Sahana_slates"
]

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

pitch_tracks = {}
for t in track_names:
    if not "2018_11_19" in t:
        p_path = f'/Volumes/MyPassport/cae-invar/data/pitch_tracks/alapana/{t}.csv'
        tonic = get_tonic(t, metadata)
        pitch, time, timestep = get_timeseries(p_path)
        pitch = pitch_seq_to_cents(pitch, tonic=tonic)
        pitch[pitch==None]=0
        pitch = interpolate_below_length(pitch, 0, (350*0.001/timestep))
        pitch_d, time_d = get_derivative(pitch, time)
        pitch_tracks[t] = (gaussian_filter1d(pitch, 2.5), time, timestep, gaussian_filter1d(pitch_d, 2.5), time_d)

all_patts = pd.read_csv(os.path.join(out_dir, 'all_groups.csv'))

#all_distances = pd.DataFrame(columns=['index1', 'index2', 'path1_start', 'path1_end', 'path2_start', 'path2_end', 'path_length', 'dtw_distance', 'dtw_distance_norm'])


## Look at pairs

# good pairs
i1 = 0
i2 = 1

# bad pair
i1 = 579
i2 = 763



i1 = 773
i2 = 1

# Bad pair: 851 and 964
row = all_patts[all_patts['index']==i1].iloc[0]
rrow = all_patts[all_patts['index']==i2].iloc[0]

qstart = row.start
qend = row.end
qtrack = row.track
qi = row['index']
(qpitch, qtime, qtimestep, qpitch_d, qtime_d) = pitch_tracks[qtrack]

sq1 = int(qstart/qtimestep)
sq2 = int(qend/qtimestep)

rstart = rrow.start
rend = rrow.end
rtrack = rrow.track
rj = rrow['index']

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
r = 0.3#for r in [0.01, 0.02, 0.03, 0.04, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.3, 0.05]:
path_dtw, dtw_val = dtw_path(pat1, pat2, radius=int(l_longest*r))
l = len(path_dtw)
dtw_norm = dtw_val/l

## Plot
plt.close()

nullfmt = NullFormatter()

# definitions for the axes
left, width = 0.12, 0.60
bottom, height = 0.08, 0.60
bottom_h =  0.16 + width 
left_h = left + 0.27 
rect_plot = [left_h, bottom, width, height]
rect_x = [left_h, bottom_h, width, 0.2]
rect_y = [left, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(2, figsize=(8, 8))

axplot = plt.axes(rect_plot)
axx = plt.axes(rect_x)
axx.grid()
axy = plt.axes(rect_y)
axy.grid()
# Plot the matrix
#axplot.pcolor(acc.T,cmap=cm.gray)
axplot.plot([x[0] for x in path_dtw], [x[1] for x in path_dtw], 'black')

axplot.set_xlim((0, len(pat1)))
axplot.set_ylim((0, len(pat2)))
axplot.tick_params(axis='both', which='major', labelsize=18)

# Plot time serie horizontal
axx.plot(pat1,'.', color='k')
axx.tick_params(axis='both', which='major', labelsize=18)
xloc = plt.MaxNLocator(4)
x2Formatter = FormatStrFormatter('%d')
axx.yaxis.set_major_locator(xloc)
axx.yaxis.set_major_formatter(x2Formatter)

# Plot time serie vertical
axy.plot(pat2, range(len(pat2)),'.',color='k')
axy.invert_xaxis()
yloc = plt.MaxNLocator(4)
xFormatter = FormatStrFormatter('%d')
axy.xaxis.set_major_locator(yloc)
axy.xaxis.set_major_formatter(xFormatter)
axy.tick_params(axis='both', which='major', labelsize=18)

# Limits
axx.set_xlim(axplot.get_xlim())
axy.set_ylim(axplot.get_ylim())

plt.title(f'r={r}, dtw={round(dtw_norm,2)}')
plt.savefig(f'dtw_plot_new_r={r}.png')
plt.close()
