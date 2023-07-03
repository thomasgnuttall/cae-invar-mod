import pandas as pd
import numpy
import librosa
import os

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

# 2018_11_19 bad voice
track_names = [
    '2018_11_13_am_Sec_1_P2_Varaali_slates',
    '2018_11_13_pm_Sec_4_P1_Shankara_slates',
    '2018_11_15_Sec_9_P1_Todi_slates',
    '2018_11_18_pm_Sec_1_P1_Shankara_slates',
    '2018_11_13_am_Sec_2_P1_Shankara_slates',
    '2018_11_13_pm_Sec_5_P1_Bilahari_slates',
    '2018_11_18_am_Sec_1_P1_Varaali_slates',
    '2018_11_18_pm_Sec_1_P2_Varaali_slates',
    '2018_11_13_am_Sec_3_P1_Anandabhairavi_slates',
    '2018_11_13_pm_Sec_5_P2_Atana_slates',
    '2018_11_18_am_Sec_2_P2_Shankara_slates',
    '2018_11_18_pm_Sec_1_P3_Bilahari_slates',
    '2018_11_13_am_Sec_3_P3_Kalyani_slates',
    '2018_11_15_Sec_10_P1_Atana_slates',
    '2018_11_18_am_Sec_3_P1_Anandabhairavi_A_slates',
    '2018_11_18_pm_Sec_2_P2_Anandabhairavi_full_slates',
    '2018_11_13_am_Sec_3_P5_Todi_slates',
    '2018_11_15_Sec_12_P1_Kalyani_slates',
    '2018_11_18_am_Sec_3_P2_Anandabhairavi_B_slates',
    '2018_11_18_pm_Sec_3_P1_Kalyani_slates',
    '2018_11_13_am_Sec_3_P8_Bilahari_B_slates',
    '2018_11_15_Sec_13_P1_Bhairavi_slates',
    '2018_11_18_am_Sec_4_P1_Kalyani_slates',
    '2018_11_18_pm_Sec_4_P1_Bhairavi_slates',
    '2018_11_13_am_Sec_4_P1_Atana_slatesB',
    '2018_11_15_Sec_1_P1_Anandabhairavi_A_slates',
    '2018_11_18_am_Sec_5_P1_Bhairavi_slates',
    '2018_11_18_pm_Sec_4_P2_Atana_slates',
    '2018_11_13_pm_Sec_1_P1_Varaali_slates',
    '2018_11_15_Sec_2_P1_Anandabhairavi_B_slates',
    '2018_11_18_am_Sec_5_P3_Bilahari_slates',
    '2018_11_18_pm_Sec_5_P1_Todi_full_slates',
    '2018_11_13_pm_Sec_1_P2_Anandabhairavi_slates',
    '2018_11_15_Sec_3_P1_Anandabhairavi_C_slates',
    '2018_11_18_am_Sec_6_P1_Atana_A_slates',
    '2018_11_18_pm_Sec_5_P2_Sahana_slates',
    '2018_11_13_pm_Sec_2_P1_Kalyani_slates',
    '2018_11_15_Sec_4_P1_Shankara_slates',
    '2018_11_18_am_Sec_6_P2_Atana_B_slates',
    '2018_11_13_pm_Sec_3_P1_Todi_A_slates',
    '2018_11_15_Sec_6_P1_Varaali_slates',
    '2018_11_18_am_Sec_6_P4_Todi_slates',
    '2018_11_13_pm_Sec_3_P2_Todi_B_slates',
    '2018_11_15_Sec_8_P1_Bilahari_slates',
    '2018_11_18_am_Sec_6_P6_Bilahari_B'
]

#track_names = ['2018_11_13_am_Sec_3_P1_Anandabhairavi_V.2', '2018_11_13_pm_Sec_3_P1_Todi_A', '2018_11_15_Sec_6_P1_Varaali', '2018_11_18_am_Sec_6_P6_Bilahari_B', '2018_11_19_Sec2_P1_Kalyani', '2018_11_18_pm_Sec1_P3_Bilahari']

# Features Extraction
import csv
from exploration.io import create_if_not_exists
from exploration.pitch import silence_stability_from_file
from compiam.melody.pitch_extraction import Melodia
from compiam.melody.pattern.sancara_search.extraction.pitch import interpolate_below_length
from compiam import load_model
import numpy as np

model = load_model('melody:ftanet-carnatic')

all_paths = []
# pitch tracks
failed_tracks = []
for t in track_names:
    print(t)
    print('-'*len(t))
    spleeter_path = f'audio/lara_wim2/spleeter/{t}.mp3'
    pitch_path = f'data/pitch_tracks/alapana/{t}.csv'
    stab_path = f'data/stability_tracks/alapana/{t}.csv'

    tonic = tonic_dict[t]
    raga = get_raga(t, metadata)

    if not os.path.exists(pitch_path):
        create_if_not_exists(pitch_path)
        create_if_not_exists(stab_path)
        
        print('- extracting pitch')
        prediction = model.predict(spleeter_path)
        pitch = prediction[:,1]
        pitch[np.where(pitch<80)[0]]=0
        prediction[:,1] = pitch

        # pitch track
        with open(pitch_path, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            for row in prediction:
                # write a row to the csv file
                writer.writerow(row)
        
        print('- extracting stability mask')
        silence_stability_from_file(pitch_path, stab_path, tonic=tonic, freq_var_thresh_stab=60, gap_interp=0.350)
    else:
        print(f'{t} already exists!')
    this_path = ((t, raga, tonic), (spleeter_path, stab_path, pitch_path))
    all_paths.append(this_path)


import skimage.io

from convert import *
from exploration.pitch import *
from exploration.io import *

run_keyword= 'alapana_spleet'

cache_dir = "cache"
cuda = False
train_it = True
continue_train = False
start_epoch = 0
test_data = 'jku_input.txt'
test_cqt = False

data_type = 'cqt'

# train params
samples_epoch = 100000
batch_size = 1000
epochs = 1000
lr = 1e-3
sparsity_reg = 0e-5
weight_reg = 0e-5
norm_loss = 0
equal_norm_loss = 0
learn_norm = False
set_to_norm = -1
power_loss = 1
seed = 1
plot_interval = 500

# model params
dropout = 0.5
n_bases = 256

# CQT params
length_ngram = 32
fmin = 65.4
hop_length = 1984
block_size = 2556416
n_bins = 120
bins_per_oct = 24
sr = 44100

# MIDI params
min_pitch = 40
max_pitch = 100
beats_per_timestep = 0.25

# data loader
emph_onset = 0
rebuild = False
# shiftx (time), shifty (pitch)
shifts = 12, 24
# scalex, scaley (scaley not implemented!)
scales = 0, 0
# transform: 0 -> shifty (pitch), 1 -> shiftx (time), 2 -> scalex (time)
transform = 0, 1


torch.manual_seed(seed)
np.random.seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
out_dir = os.path.join("output", run_keyword)

assert os.path.exists(out_dir), f"The output directory {out_dir} does " \
    f"not exist. Did you forget to train using the run_keyword " \
    f"{run_keyword}?"

if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

if data_type == 'cqt':
    in_size = n_bins * length_ngram
else:
    raise AttributeError(f"Data_type {data_type} not supported. "
                         f"Possible type is 'cqt'.")
model = Complex(in_size, n_bases, dropout=dropout)
model_save_fn = os.path.join(out_dir, "model_complex_auto_"
                                     f"{data_type}.save")
model.load_state_dict(torch.load(model_save_fn, map_location='cpu'), strict=False)


def find_nearest(array, value, ix=True):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if ix:
        return idx
    else:
        return array[idx]


# params from config
length_ngram=32
cuda=False
data_type = 'cqt'
n_bins=120
bins_per_oct=24
fmin=80
hop_length=1984
step_size=1
mode='cosine'


for i_path in range(len(all_paths)):
    try:
        (title, raga, tonic), (file, mask_file, pitch_file) = all_paths[i_path]
        track_name = pitch_file.replace('.csv','').split('/')[-1]
        out_dir = f'/Volumes/Shruti/asplab2/cae-invar/data/self_similarity/{track_name}/'
        if os.path.isfile(os.path.join(out_dir,'self_sim.npy')):
            print(f'Skipping {track_name}')
        else:
            print('\n---------')
            print(title)
            print('---------')

            #def create_matrices(file, mask_file, length_ngram, cuda, data_type, n_bins, bins_per_oct, fmin, hop_length, step_size=1, mode='cosine'):
            print('Computing CAE features')
            data = get_input_repr(file, data_type, n_bins, bins_per_oct, fmin, hop_length)
            mask, time, timestep  = get_timeseries(mask_file)
            pitch, fr_time, fr_timestep  = get_timeseries(pitch_file)

            ampls, phases = to_amp_phase(model, data, step_size=step_size, length_ngram=length_ngram, cuda=cuda)
            results = np.array([ampls.detach().numpy(), phases.detach().numpy()])

            print('Computing self similarity matrix')
            # get mask of silent and stable regions
            matrix_mask = []
            for i in range(ampls.shape[0]):
                t = (i+1)*hop_length/sr
                ix = find_nearest(time, t)
                matrix_mask.append(mask[ix])
            matrix_mask = np.array(matrix_mask)

            good_ix = np.where(matrix_mask==0)[0]
            orig_sparse_lookup = {g:s for s,g in enumerate(good_ix)}
            sparse_orig_lookup = {s:g for g,s in orig_sparse_lookup.items()}
            boundaries_orig = []
            for i in range(1, len(matrix_mask)):
                curr = matrix_mask[i]
                prev = matrix_mask[i-1]
                if curr==0 and prev==1:
                    boundaries_orig.append(i)
                elif curr==1 and prev==0:
                    boundaries_orig.append(i-1)
            boundaries_sparse = np.array([orig_sparse_lookup[i] for i in boundaries_orig])
            # boundaries contain two consecutive boundaries for each gap
            # but not if the excluded region leads to the end of the track
            red_boundaries_sparse = []
            boundaries_mask = [0]*len(boundaries_sparse)
            for i in range(len(boundaries_sparse)):
                if i==0:
                    red_boundaries_sparse.append(boundaries_sparse[i])
                    boundaries_mask[i]=1
                if boundaries_mask[i]==1:
                    continue
                curr = boundaries_sparse[i]
                prev = boundaries_sparse[i-1]
                if curr-prev == 1:
                    red_boundaries_sparse.append(prev)
                    boundaries_mask[i]=1
                    boundaries_mask[i-1]=1
                else:
                    red_boundaries_sparse.append(curr)
                    boundaries_mask[i]=1
            boundaries_sparse = np.array(sorted(list(set(red_boundaries_sparse))))

            sparse_ampls = ampls[good_ix]
            matrix_orig = create_ss_matrix(sparse_ampls, mode=mode)

            print('Normalising self similarity matrix')
            matrix = 1 / (matrix_orig + 1e-6)

            for k in range(-8, 9):
                eye = 1 - np.eye(*matrix.shape, k=k)
                matrix = matrix * eye

            flength = 10
            ey = np.eye(flength) + np.eye(flength, k=1) + np.eye(flength, k=-1)
            matrix = convolve2d(matrix, ey, mode="same")

            diag_mask = np.ones(matrix.shape)
            diag_mask = (diag_mask - np.diag(np.ones(matrix.shape[0]))).astype(np.bool)

            mat_min = np.min(matrix[diag_mask])
            mat_max = np.max(matrix[diag_mask])
            
            matrix[~diag_mask] = 0

            matrix -= matrix.min()
            matrix /= (matrix.max() + 1e-8)

            #for b in boundaries_sparse:
            #    matrix[:,b] = 1
            #    matrix[b,:] = 1

            #plt.imsave('random/4final.png', matrix, cmap="hot")

            ## Output
            metadata = {
                'orig_size': (len(data), len(data)),
                'sparse_size': (matrix.shape[0], matrix.shape[0]),
                'orig_sparse_lookup': orig_sparse_lookup,
                'sparse_orig_lookup': sparse_orig_lookup,
                'boundaries_orig': boundaries_orig,
                'boundaries_sparse': boundaries_sparse,
                'audio_path': file,
                'pitch_path': pitch_file,
                'stability_path': mask_file,
                'raga': raga,
                'tonic': tonic,
                'title': title
            }

            out_path_mat = os.path.join(out_dir, 'self_sim.npy')
            out_path_meta = os.path.join(out_dir, 'metadata.pkl')
            out_path_feat = os.path.join(out_dir, "features.pyc.bz")

            create_if_not_exists(out_dir)

            print(f"Saving features to {out_path_feat}..")
            save_pyc_bz(results, out_path_feat)

            print(f"Saving self sim matrix to {out_path_mat}..")
            np.save(out_path_mat, matrix)

            print(f'Saving metadata to {out_path_meta}')
            write_pkl(metadata, out_path_meta)
    except Exception as e:
        print(f'{title} failed')
        print(f'{e}')


from experiments.alapana_dataset_analysis.main import main

import faulthandler

faulthandler.enable()
sr = 44100
cqt_window = 1984
s1 = None
s2 = None
gap_interp = 0.35
stab_hop_secs = 0.2
min_stability_length_secs = 1.0
freq_var_thresh_stab = 60
conv_filter_str = 'sobel'
gauss_sigma = None
cont_thresh = 0.15
etc_kernel_size = 10
binop_dim = 3
min_diff_trav = 0.5 #0.1
min_in_group = 2
match_tol = 1
ext_mask_tol = 0.5
n_dtw = 10
thresh_cos = None
top_n = 1000
write_plots = True
write_audio = True
write_patterns = True
write_annotations = False
partial_perc = 0.66
perc_tail = 0.5
plot=False
min_pattern_length_seconds = 1.8

group_len_var = 0.5 # Current Best: 1
thresh_dtw = 4.5 # Current Best: 8
dupl_perc_overlap_intra = 0.6 # Current Best: 0.6
dupl_perc_overlap_inter = 0.75 # Current Best: 0.75

i=0
run_name='test'
allbt = [0.005, 0.0025, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.115, 0.125, 0.13, 0.135, 0.14]
allbt.reverse()
for bin_thresh in allbt:
    if dupl_perc_overlap_intra > dupl_perc_overlap_inter:
        continue
    for track_name in track_names:#['2018_11_13_am_Sec_3_P1_Anandabhairavi_V.2']:#, '2018_11_13_pm_Sec_3_P1_Todi_A', '2018_11_15_Sec_6_P1_Varaali', '2018_11_18_am_Sec_6_P6_Bilahari_B', '2018_11_19_Sec2_P1_Kalyani', '2018_11_18_pm_Sec1_P3_Bilahari']:
        i+=1
        run_name = f"bin_thresh={bin_thresh}" #_group_len_var={group_len_var}_dintra={dupl_perc_overlap_intra}_dinter={dupl_perc_overlap_inter}"
        print(f'{i}/360: {run_name}')
        bin_thresh_segment = bin_thresh*0.75
        main(
            track_name, run_name, sr, cqt_window, s1, s2,
            gap_interp, stab_hop_secs, min_stability_length_secs,
            60, conv_filter_str, bin_thresh,
            bin_thresh_segment, perc_tail,  gauss_sigma, cont_thresh,
            etc_kernel_size, binop_dim, min_diff_trav, min_pattern_length_seconds,
            min_in_group, match_tol, ext_mask_tol, n_dtw, thresh_dtw, thresh_cos, group_len_var, 
            dupl_perc_overlap_inter, dupl_perc_overlap_intra, None,
            None, partial_perc, top_n, write_plots,
            write_audio, write_patterns, False, plot=False)

