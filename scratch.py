%load_ext autoreload
%autoreload 2

import os

import skimage.io

from complex_auto.motives_extractor import *
from complex_auto.motives_extractor.extractor import *

from exploration.pitch import extract_pitch_track
from exploration.img import remove_diagonal, convolve_array_tile, binarize, diagonal_gaussian, hough_transform, hough_transform_new
from exploration.segments import get_all_segments, break_all_segments, do_patterns_overlap, reduce_duplicates, remove_short, group_segments
from exploration.sequence import apply_exclusions, contains_silence, min_gap, too_stable, convert_seqs_to_timestep, get_stability_mask, add_center_to_mask
from exploration.evaluation import load_annotations_new, evaluate_all_tiers
from exploration.visualisation import plot_all_sequences, plot_pitch
from exploration.io import load_sim_matrix, write_all_sequence_audio, load_yaml
from exploration.pitch import cents_to_pitch, pitch_seq_to_cents, pitch_to_cents


def remove_below_length(starts_seq, lengths_seq, timestep, min_length):
    starts_seq_long = []
    lengths_seq_long = []
    for i, group in enumerate(lengths_seq):
        this_group_l = []
        this_group_s = []
        for j,l in enumerate(group):
            if l >= min_length/timestep:
                this_group_l.append(l)
                this_group_s.append(starts_seq[i][j])
        if this_group_s:
            starts_seq_long.append(this_group_s)
            lengths_seq_long.append(this_group_l)

    return starts_seq_long, lengths_seq_long


################
## Parameters ##
################
# Output paths of each step in pipeline
out_dir = os.path.join("output", 'hpc')

sim_path = 'output/hpc/Koti Janmani.multitrack-vocal.mp3.npy'

# Sample rate of audio
sr = 44100

# size in frames of cqt window from convolution model
cqt_window = 1984

# Take sample of data, set to None to use all data
s1 = None # lower bound index
s2 = None # higher bound index

# pitch track extraction
frameSize = 2048 # For Melodia pitch extraction
hopSize = 128 # For Melodia pitch extraction
gap_interp = 250*0.001 # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]
smooth = 7 # sigma for gaussian smoothing of pitch track [set to None to skip]
audio_path_vocal = 'audio/Akkarai Sisters at Arkay by Akkarai Sisters/Koti Janmani/Koti Janmani.multitrack-vocal.mp3'
audio_path = "audio/full_dataset/Akkarai_Sisters_at_Arkay/Akkarai Sisters - Akkarai Sisters at Arkay/1 - 3 - Akkarai Sisters - Koti Janmani.mp3"

# stability identification
stab_hop_secs = 0.2 # window size for stab computations in seconds
min_stability_length_secs = 1.0 # minimum legnth of region to be considered stable in seconds
freq_var_thresh_stab = 10 # max variation in pitch to be considered stable region

# Binarize raw sim array 0/1 below and above this value...
bin_thresh = 0.6

# Gaussian filter along diagonals with sigma...
gauss_sigma = 18

# After gaussian, re-binarize with this threshold
cont_thresh = 0.15

# Hough transform parameters
min_dist_sec = 0 # min dist in seconds between lines
hough_threshold = 25

# Only search for lines between these angles (45 corresponds to main diagonal)
hough_high_angle = 45.01
hough_low_angle = 44.99

# Distance between consecutive diagonals to be joined in seconds
min_diff_trav = 0.5

# Two segments must overlap in both x and y by <dupl_perc_overlap> 
# to be considered the same, only the longest is returned
dupl_perc_overlap = 0.5

# Grouping diagonals
min_pattern_length_seconds = 1
min_length_cqt = min_pattern_length_seconds*sr/cqt_window
min_in_group = 1 # minimum number of patterns to be included in pattern group

# Minimum distance between start points (x,y) to be joined a segment group
same_seqs_thresh_secs = 0.5
same_seqs_thresh = int(same_seqs_thresh_secs*sr/cqt_window)

# Exclusions
exclusion_functions = [contains_silence, min_gap]

# Evaluation
annotations_path = 'annotations/koti_janmani.txt'
eval_tol = 0.5 # how much leniancy on each side of an annotated pattern before considering it a match (seconds)


# Output
svara_cent_path = "conf/svara_cents.yaml"
svara_freq_path = "conf/svara_lookup.yaml"

tonic = 195.99

svara_cent = load_yaml(svara_cent_path)
svara_freq = load_yaml(svara_freq_path)

yticks_dict = {k:cents_to_pitch(v, tonic) for k,v in svara_cent.items()}
yticks_dict = {k:v for k,v in yticks_dict.items() if any([x in k for x in ['S', 'R2', 'G2', 'M1', 'P', 'D2', 'N2', 'S']])}

plot_kwargs = {
    'yticks_dict':yticks_dict,
    'cents':True,
    'tonic':195.997718,
    'emphasize':['S', 'S^'],
    'figsize':(15,4)
}

# limit the number of groups outputted
top_n = 1000


####################
## Load sim array ##
####################
# Get similarity Martrix
print(f'Loading sim matrix from {sim_path}')
X = load_sim_matrix(sim_path)

# Sample for development
if all([s1,s2]):
    save_imgs = s2-s1 <= 4000
    X_samp = X.copy()[s1:s2,s1:s2]
else:
    save_imgs = False
    X_samp = X.copy()

sim_filename = os.path.join(out_dir, 'Koti Janmani_simsave.png') if save_imgs else None
gauss_filename = os.path.join(out_dir, 'Koti Janmani_gauss.png') if save_imgs else None
edge_filename = os.path.join(out_dir, 'Koti Janmani_edges.png') if save_imgs else None
bin_filename = os.path.join(out_dir, 'Koti Janmani_binary.png') if save_imgs else None
cont_filename = os.path.join(out_dir, 'Koti Janmani_cont.png') if save_imgs else None
hough_filename = os.path.join(out_dir, 'Koti Janmani_hough.png') if save_imgs else None
conv_filename = os.path.join(out_dir, 'Koti Janmani_conv.png') if save_imgs else None
diag_filename = os.path.join(out_dir, 'Koti Janmani_diag.png') if save_imgs else None

if save_imgs:
    skimage.io.imsave(sim_filename, X_samp)

##############
## Pipeline ##
##############
print('Extracting pitch track')
pitch, raw_pitch, timestep, time = extract_pitch_track(audio_path_vocal, frameSize, hopSize, gap_interp, smooth, sr)

print('Computing stability/silence mask')
stable_mask = get_stability_mask(raw_pitch, min_stability_length_secs, stab_hop_secs, freq_var_thresh_stab, timestep)
silence_mask = raw_pitch == 0
silence_mask = add_center_to_mask(silence_mask)

print('Convolving similarity matrix')
X_conv = convolve_array_tile(X_samp)

if save_imgs:
    skimage.io.imsave(conv_filename, X_conv)

print('Binarizing convolved array')
X_bin = binarize(X_conv, bin_thresh, filename=bin_filename)

print('Removing Diagonal')
X_diag = remove_diagonal(X_bin)

if save_imgs:
    skimage.io.imsave(diag_filename, X_diag)

print('Applying diagonal gaussian filter')
X_gauss = X_diag #diagonal_gaussian(X_bin, gauss_sigma, filename=gauss_filename)

print('Binarize gaussian blurred similarity matrix')
X_cont = X_gauss #binarize(X_gauss, cont_thresh, filename=cont_filename)

print('Applying Hough Transform')
peaks = hough_transform_new(X_cont, hough_high_angle, hough_low_angle, hough_threshold, filename=hough_filename)

print('Extracting segments along Hough lines')
# Format - [[(x,y), (x1,y1)],...]
all_segments = get_all_segments(X_cont, peaks, min_diff_trav, min_length_cqt, cqt_window, sr)

print('Breaking segments with stable regions')
# Format - [[(x,y), (x1,y1)],...]
all_broken_segments = break_all_segments(all_segments, stable_mask, cqt_window, sr, timestep)

print('Breaking segments with silent regions')
# Format - [[(x,y), (x1,y1)],...]
all_broken_segments = break_all_segments(all_broken_segments, silence_mask, cqt_window, sr, timestep)

print('Reducing Segments')
# The Hough transform allows for the same segment to be intersected
# twice by lines of slightly different angle. We want to take the 
# longest of these duplicates and discard the rest
all_segments_reduced = reduce_duplicates(all_broken_segments, perc_overlap=dupl_perc_overlap)
all_segments_reduced = remove_short(all_segments_reduced, min_length_cqt)

print('Grouping Segments')
all_groups = group_segments(all_segments_reduced, perc_overlap=dupl_perc_overlap)

print('Convert sequences to pitch track timesteps')
starts_seq, lengths_seq = convert_seqs_to_timestep(all_groups, cqt_window, sr, timestep)

print('Applying exclusion functions')
#starts_seq_exc, lengths_seq_exc = apply_exclusions(raw_pitch, starts_seq, lengths_seq, exclusion_functions, min_in_group)
starts_seq_exc,  lengths_seq_exc = remove_below_length(starts_seq, lengths_seq, timestep, min_pattern_length_seconds)

starts_sec_exc = [[x*timestep for x in p] for p in starts_seq_exc]
lengths_sec_exc = [[x*timestep for x in l] for l in lengths_seq_exc]

print('Evaluating')
annotations_orig = load_annotations_new(annotations_path)
metrics = evaluate_all_tiers(annotations_orig, starts_sec_exc, lengths_sec_exc, eval_tol)

# all_recalls = []
# all_evals = [0.05*i for i in range(int(10/0.05))]
# for e in all_evals:
#     these_metrics = evaluate_all_tiers(annotations_orig, starts_sec_exc, lengths_sec_exc, e)
#     all_recalls.append(these_metrics['full_match_recall_all'])

# plt.figure(figsize=(10,5))
# plt.plot(all_evals, all_recalls)
# plt.title('Performance with varying evaluation tolerance')
# plt.xlabel('Evaluation tolerance')
# plt.ylabel('Recall for all patterns')
# plt.grid()
# plt.savefig('images/eval_tol_experiment.png')
# plt.close('all')


############
## Output ##
############
print('Writing all sequences')
plot_all_sequences(raw_pitch, time, lengths_seq_exc[:top_n], starts_seq_exc[:top_n], 'output/new_hough', clear_dir=True, plot_kwargs=plot_kwargs)
write_all_sequence_audio(audio_path, starts_seq_exc[:top_n], lengths_seq_exc[:top_n], timestep, 'output/new_hough')




#####################################################
## Plotting annotations and Results on Sim Matrix  ##
#####################################################
def add_line_to_plot(arr, x0, x1, y0, y1):
    
    length = int(np.hypot(x1-x0, y1-y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
    x = x.astype(int)
    y = y.astype(int)

    arr[x, y] = 100

    return arr


def get_lines(s1, s2):
    n = len(s1)
    all_lines = []
    for i in range(n):
        x0 = s1[i]
        x1 = s2[i]
        for j in range(n):
            if j==i:
                continue
            y0 = s1[j]
            y1 = s2[j]
            all_lines.append((x0, x1, y0, y1))
    return all_lines


def add_annotations_to_plot(arr, annotations, sr, cqt_window):
    arr_ = arr.copy()
    annotations_grouped = annotations_orig.groupby('text')['s1']\
                                          .apply(list)\
                                          .reset_index()

    annotations_grouped['s2'] = annotations_orig.groupby('text')['s2']\
                                                .apply(list)\
                                                .reset_index()['s2']
    
    for i, row in annotations_grouped.iterrows():
        s1 = row['s1']
        s2 = row['s2']
        these_lines = get_lines(s1, s2)
        for x0, x1, y0, y1 in these_lines:
            arr_ = add_line_to_plot(
                arr_, int(x0*sr/cqt_window), int(x1*sr/cqt_window), 
                int(y0*sr/cqt_window), int(y1*sr/cqt_window))

    return arr_


def add_patterns_to_plot(arr, patterns, lengths, sr, cqt_window):
    arr_ = arr.copy()
    
    for i, group in enumerate(patterns):
        lens = lengths[i]
        
        s1 = group
        s2 = [g+lens[j] for j,g in enumerate(s1)]

        these_lines = get_lines(s1, s2)
        for x0, x1, y0, y1 in these_lines:
            arr_ = add_line_to_plot(
                arr_, int(x0*sr/cqt_window), int(x1*sr/cqt_window), 
                int(y0*sr/cqt_window), int(y1*sr/cqt_window))

    return arr_


def add_segments_to_plot(arr, segments):
    arr_ = arr.copy()
    for (x0, y0), (x1, y1) in segments:
        arr_ = add_line_to_plot(arr_, int(x0), int(x1), int(y0), int(y1))
    return arr_


X_canvas = X_samp.copy()
X_canvas[:] = 0

# Orig matrix
X_orig = X.copy()[samp1:samp2,samp1:samp2]
skimage.io.imsave('images/sim_mat.png', X_orig)

# Annotations
X_annotate = add_annotations_to_plot(X_canvas, annotations_orig, sr, cqt_window)
X_annotate_samp = X_annotate.copy()[samp1:samp2,samp1:samp2]
skimage.io.imsave('images/annotations_sim_mat.png', X_annotate_samp)

# Found segments from image processing
X_segments = add_segments_to_plot(X_canvas, all_segments)
X_segments_samp = X_segments.copy()[samp1:samp2,samp1:samp2]
skimage.io.imsave('images/segments_sim_mat.png', X_segments_samp)

# Found segments broken from image processing
X_segments = add_segments_to_plot(X_canvas, all_segments_reduced)
X_segments_samp = X_segments.copy()[samp1:samp2,samp1:samp2]
skimage.io.imsave('images/segments_broken_sim_mat.png', X_segments_samp)

# Patterns from full pipeline
X_patterns = add_patterns_to_plot(X_canvas, [starts_sec_exc[0]], [lengths_sec_exc[0]], sr, cqt_window)
X_patterns_samp = X_patterns.copy()[samp1:samp2,samp1:samp2]
skimage.io.imsave('images/patterns_sim_mat.png', X_patterns_samp)














###########################
## All Patterns Grouping ##
###########################
import itertools

import fastdtw
from scipy.spatial.distance import euclidean
import tqdm

dtw_radius_frac = 45

all_seq_separated = [x for y in starts_seq_exc for x in y]
all_len_separated = [x for y in lengths_seq_exc for x in y]
all_indices = list(range(len(all_seq_separated)))

all_seq_dtw = pd.DataFrame(columns=['i1', 'i2', 'dtw', 'cos', 'cos_recip', 'cos_zero', 'cos_zero_recip', 'dtw_min_length', 'len_seq1_dtw', 'len_seq2_dtw', 'len_cos'])

for i1, i2 in tqdm.tqdm(list(itertools.combinations(all_indices, 2))):
    
    # DTW From pitch track
    s1 = all_seq_separated[i1]
    s2 = all_seq_separated[i2]
    
    l1 = all_len_separated[i1]
    l2 = all_len_separated[i2]

    seq1 = pitch[s1:s1+l1]
    seq2 = pitch[s2:s2+l2]

    min_length = min([len(seq1), len(seq2)])

    dtw = fastdtw.fastdtw(seq1, seq2, radius=int(min_length/dtw_radius_frac), dist=euclidean)[0]/min_length

    # Cosine from similarity matrix
    scqt1 = int(s1*(sr*timestep)/cqt_window)
    scqt2 = int(s2*(sr*timestep)/cqt_window)
    
    lcqt1 = int(l1*(sr*timestep)/cqt_window)
    lcqt2 = int(l2*(sr*timestep)/cqt_window)
    
    x0 = scqt1
    y0 = scqt2
    x1 = scqt1 + lcqt1
    y1 = scqt2 + lcqt2

    length = int(np.hypot(x1-x0, y1-y0))

    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
    x = x.astype(int)
    y = y.astype(int)

    # Extract the values along the line
    zi = X[x, y]

    # X stores reciprocal of the cosine distance
    cos = np.mean(1/zi)
    cos_recip = np.mean(zi)
    
    zi[zi<0] = 0

    cos_zero = np.mean(1/zi)
    cos_zero_recip = np.mean(zi)

    row = {
        'i1':i1,
        'i2':i2,
        'dtw':dtw,
        'cos_recip':cos_recip,
        'cos':cos,
        'cos_zero_recip':cos_zero_recip,
        'cos_zero':cos_zero,
        'dtw_min_length':min_length,
        'len_seq1_dtw':len(seq1),
        'len_seq2_dtw':len(seq2),
        'len_cos':len(zi)
    }

    all_seq_dtw = all_seq_dtw.append(row, ignore_index=True)

# add self similarity
for i in all_indices:
    row = {
        'i1':i,
        'i2':i,
        'dtw':0,
        'cos_recip':np.Inf,
        'cos':0,
        'cos_zero_recip':np.Inf,
        'cos_zero':0,
        'dtw_min_length':all_len_separated[i],
        'len_seq1_dtw':all_len_separated[i],
        'len_seq2_dtw':all_len_separated[i],
        'len_cos':all_len_separated[i]
    }

    all_seq_dtw = all_seq_dtw.append(row, ignore_index=True)

all_seq_dtw.to_csv('results_tables/new_model_dtw_all_pairs.csv', index=False)

# Similarity Distribution Plots
plt.hist(all_seq_dtw['dtw'].values, bins=500, color='darkgreen')
plt.title('Distribution of inter-sequence DTW')
plt.xlabel('DTW bin')
plt.ylabel('Population')
plt.savefig('images/dtw_histogram.png')
plt.close('all')


for_plot = all_seq_dtw[all_seq_dtw['cos_zero']!=np.Inf]
plt.hist(for_plot['cos_zero'].values, bins=250, color='darkgreen')
plt.title('Distribution of inter-sequence cosine distance')
plt.xlabel('Cosine distance bin')
plt.ylabel('Population')
plt.savefig('images/cos_histogram.png')
plt.close('all')


# Clustering
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as hcluster

distance = all_seq_dtw\
            .pivot(index='i1', columns='i2', values='dtw')\
            .fillna(0)

data = distance.values

for i in range(data.shape[0]):
    for j in range(i, data.shape[0]):
        data[j][i] = data[i][j]

distVec = ssd.squareform(data)
linkage = hcluster.linkage(distVec, method='ward')

clustering = hcluster.cut_tree(linkage, n_clusters=range(len(linkage)))

from scipy.spatial.distance import euclidean

def DaviesBouldin(X, labels):
    n_cluster = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_cluster)]
    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    variances = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    db = []

    for i in range(n_cluster):
        for j in range(n_cluster):
            if j != i:
                db.append((variances[i] + variances[j]) / euclidean(centroids[i], centroids[j]))

    return(np.max(db) / n_cluster)


def evaluate(disance, clustering_results, k_min, k_max):
    X = distance.values
    return [DaviesBouldin(X, clustering_results[:,i]) \
            for i in range(len(clustering_results))[k_min:k_max]]

k_min = 2
k_max = 100
evaluation = evaluate(distance, clustering, k_min, k_max)

from kneed import KneeLocator

x = list(range(k_min, k_max))
knee = KneeLocator(x, evaluation, S=0.4, curve="convex", direction="decreasing").knee

plt.figure(figsize=(12,5))
plt.plot(x, evaluation,color='darkgreen')
plt.xticks(np.arange(min(x), max(x)+1, 2.0),size=8)
plt.axvline(knee, linestyle='--', color='darkred', linewidth=0.7)
plt.title('Davies Bouldin Score for n clusters')
plt.xlabel('Number of Clusters, n')
plt.ylabel('DBS')
plt.grid()
plt.savefig('images/DaviesBouldin.png')
plt.close('all')



## Best model
n = 24

cluster_seqs = {}
cluster_lens = {}
for ix,c in enumerate(clustering[:,n]):
    if c in cluster_seqs:
        cluster_seqs[c].append(all_seq_separated[ix])
        cluster_lens[c].append(all_len_separated[ix])
    else:
        cluster_seqs[c] = [all_seq_separated[ix]]
        cluster_lens[c] = [all_len_separated[ix]]

cluster_seqs = [v for k,v in cluster_seqs.items()]
cluster_lens = [v for k,v in cluster_lens.items()]





plot_all_sequences(raw_pitch, time, cluster_lens[:top_n], cluster_seqs[:top_n], 'output/clustering', clear_dir=True, plot_kwargs=plot_kwargs)
write_all_sequence_audio(audio_path, cluster_seqs[:top_n], cluster_lens[:top_n], timestep, 'output/clustering')













############################
# Plot individual sequence #
############################
from exploration.visualisation import plot_subsequence_w_stability

sp  = 5300
l = 1000
plot_subsequence_w_stability(sp, l, raw_pitch, time, stable_mask, timestep, path='images/stab_check.png', plot_kwargs=plot_kwargs)

sp  = x_start_ts
l = x_end_ts - x_start_ts
plot_subsequence_w_stability(sp, l, raw_pitch, time, stable_mask, timestep, path='images/seqx_stab.png', plot_kwargs=plot_kwargs)


sp  = y_start_ts
l = y_end_ts - y_start_ts
plot_subsequence_w_stability(sp, l, raw_pitch, time, stable_mask, timestep, path='images/seqy_stab.png', plot_kwargs=plot_kwargs)





############
# Database #
############
from exploration.utils import sql
from credentials import settings
import psycopg2

def insertResults(records, params):
    try:
        connection = psycopg2.connect(**settings)

        cursor = connection.cursor()

        # Update single record now
        sql_insert_query = """ 
        INSERT INTO results 
        (patternnumber, recordingid, elementnumber, durationelements, starttimeseconds, durationseconds, patterngroup, rankingroup)
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.executemany(sql_insert_query, records)
        connection.commit()
        count = cursor.rowcount
        print(count, "Record Updated successfully ")

    except (Exception, psycopg2.Error) as error:
        print("Error in update operation", error)

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

def insertSimilarity(records, params):
    try:
        connection = psycopg2.connect(**settings)

        cursor = connection.cursor()

        # Update single record now
        sql_insert_query = """ 
        INSERT INTO similarity 
        (patternnumberone, patternnumbertwo, similarityname, similarity)
        VALUES(%s, %s, %s, %s)"""
        cursor.executemany(sql_insert_query, records)
        connection.commit()
        count = cursor.rowcount
        print(count, "Record Updated successfully ")

    except (Exception, psycopg2.Error) as error:
        print("Error in update operation", error)

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

recording_id = 'brovabarama'
records = []

pattern_num = 0
pattern_num_lookup = {}
for i, seq in enumerate(starts_seq_cut):
    for j, s in enumerate(seq):
        length = lengths_seq_cut[i][j]
        length_secs = round(length*timestep,2)
        start_time_secs = round(s*timestep,2)
        records.append((pattern_num, recording_id, s, length, start_time_secs, length_secs, i, j))
        pattern_num_lookup[pattern_num] = (i,j)
        pattern_num += 1


insertTable(records, settings)



import itertools
similarities = []
for s1, s2 in itertools.combinations(pattern_num_lookup.keys(), 2):
    for n in ['cosine', 'dtw', 'eucliedean']:
        similarities.append((s1, s2, n, np.random.random()))





# train model more
    # - parameters
        # Tune frequency bands 
            # for this music, perhaps a standard fourier transform would work better?
            # what is fmin
            # how many octaves
            # frequency distribution across all tracks can inform parameters
    # - check graphs
    # - no further test performance increase after ~1250 epochs

# link features to annotations from Lara for phrase onset detection
    # load features and annotations

from complex_auto.util import load_pyc_bz
import textgrid
import pandas as pd

import math

def load_annotations(path):
    """
    Load text grid annotations from <path>
    return pandas df
    """
    tg = textgrid.TextGrid.fromFile(path)

    df = pd.DataFrame(columns=['tier','s1', 's2', 'text'])
    for tier in tg:
        name = tier.name
        intervals = tier.intervals
        for i in intervals:
            d = {
                'tier':name,
                's1': i.minTime,
                's2': i.maxTime,
                'text': i.mark
            }
            df = df.append(d, ignore_index=True)
    return df


def transform_features(features):
    amp_arr = features[0].detach().numpy()
    phase_arr = features[1].detach().numpy()
    
    nbins = amp_arr.shape[1]
    
    amp_cols = [f'amp_{i}' for i in range(nbins)]
    phase_cols = [f'phase_{i}' for i in range(nbins)]
    
    amp_df = pd.DataFrame(amp_arr, columns=amp_cols)
    phase_df = pd.DataFrame(phase_arr, columns=phase_cols)

    df = pd.concat([amp_df, phase_df], axis=1)
    df['window_num'] = df.index
    return df


def second_to_window(onset, sr, hop_size):
    onset_el = onset*sr
    window_num = math.floor(onset_el/hop_size)
    return window_num


features_paths = [
    'output/hpc/Koti Janmani.multitrack-vocal.mp3_repres.pyc.bz',
    'output/hpc/Shankari Shankuru.multitrack-vocal.mp3_repres.pyc.bz',
    'output/hpc/Sharanu Janakana.multitrack-vocal.mp3_repres.pyc.bz'
]
annotations_paths = [
    '../carnatic-motifs/Akkarai_Sisters_-_Koti_Janmani_multitrack-vocal_-_ritigowla.TextGrid',
    '../carnatic-motifs/Akkarai_Sisters_-_Shankari_Shankuru_multitrack-vocal_-_saveri.TextGrid',
    '../carnatic-motifs/Salem_Gayatri_Venkatesan_-_Sharanu_Janakana_multitrack-vocal_-_bilahari_copy.TextGrid'
]


all_features = pd.DataFrame()

for i,(fp, ap) in enumerate(zip(features_paths, annotations_paths)):

    # array of [amplitude, phase]
    features_raw = load_pyc_bz(fp)
    features = transform_features(features_raw)
    annotations = load_annotations(ap)

    hop_size = cqt_window # 1984

    annotations['window_num'] = annotations['s1'].apply(lambda y: second_to_window(y, sr, hop_size))

    features['is_onset'] = features['window_num'].isin(annotations['window_num'])
    features['is_test'] = i==2
    all_features = all_features.append(features, ignore_index=True)




# Classification
import lightgbm as lgb
from scipy.stats import randint as sp_randint
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     RandomizedSearchCV, TimeSeriesSplit,
                                     cross_val_score, train_test_split)
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

def random_float_inrange(N,a=0.005,b=0.1):
    return[((b - a) * np.random.random_sample()) + a for _ in range(N)]


#df_train, df_test = train_test_split(all_features, test_size=0.4, random_state=42)
df_train = all_features[all_features['is_test']==False]
df_test = all_features[all_features['is_test']==True]

# resample
# Resample to account for huge sparsity
pos_frame = df_train[df_train['is_onset']==1]
neg_frame = df_train[df_train['is_onset']!=1]

while sum(df_train['is_onset'])/len(df_train) < 0.3:
    print(sum(df_train['is_onset'])/len(df_train))
    random_rec = pos_frame.sample(1000)
    df_train = df_train.append(random_rec, ignore_index=True)

# shuffle frame
df_train = df_train.iloc[np.random.permutation(len(df_train))].reset_index(drop=True)


feat_names = [c for c in df_train if c not in ['is_onset', 'window_num', 'is_test']]

X_train = df_train[feat_names].values
y_train = df_train['is_onset'].values

X_test = df_test[feat_names].values
y_test = df_test['is_onset'].values


param_dist = {'reg_sqrt':[True],
             'learning_rate':[0.001,0.01,0.1, 0.5],
             'max_depth':[2,4,8,12],
             'min_data_in_leaf':[1,5,10],
             'num_leaves':[5,10,15,20,25],
             'n_estimators':[100,200,300,400],
             'colsample_bytree':[0.6, 0.75, 0.9]}


# Final features from gridsearch
final_params = {
    'colsample_bytree': 0.6463615939999198,
    'learning_rate': 0.1280212488889668,
    'max_depth': 40,
    'min_data_in_leaf': 27,
    'n_estimators': 982,
    'num_leaves': 46,
    'reg_sqrt': True
}

lgb_model = lgb.LGBMClassifier(**final_params)

# Gridsearch
lgb_model = lgb.LGBMClassifier()
lgb_model = RandomizedSearchCV(lgb_model, param_distributions=param_dist,
                               n_iter=1000, cv=3, n_jobs=-1,
                               scoring='recall', random_state=42)

lgb_model.fit(X_train, y_train)




y_pred = lgb_model.predict(X_test)
for scorer in recall_score, precision_score, f1_score, roc_auc_score:
    print(f'{scorer.__name__}: {scorer(y_test, y_pred)}')


importances = list(sorted(zip(feat_names, lgb_model.feature_importances_), key=lambda y: -y[1]))
importances[:10]





# black out similarity grid based on
    # consonant onset
    # silence
    # stability

# link db to ladylane

















sql("""

    SELECT 
    results.patternnumber,
    results.patterngroup,
    results.rankingroup,
    results.starttimeseconds,
    results.durationseconds

    FROM results

    WHERE results.recordingid = 'brovabarama'
    AND results.patterngroup = 1


""")



sql("""

    SELECT 

    patternnumberone,
    patternnumbertwo,
    similarity,
    similarityname

    FROM similarity
    WHERE similarityname = 'cosine'
    AND (patternnumberone = 4 OR patternnumbertwo = 4)

    ORDER BY similarity
    
""")



















insertSimilarity(similarities, settings)

#######################
# Output subsequences #
#######################
from exploration.visualisation import plot_all_sequences, plot_pitch
from exploration.io import write_all_sequence_audio

plot_kwargs = {
    'yticks_dict':{},
    'cents':True,
    'tonic':195.997718,
    'emphasize':{},#['S', 'S^'],
    'figsize':(15,4)
}

starts_seq_cut = [[a,c] for a,b,c,d in patterns_seq]
lengths_seq_cut = [[max([b-a, d-c])]*2 for a,b,c,d in patterns_seq]

plot_all_sequences(pitch, time, lengths_seq_cut, starts_seq_cut, out_dir, clear_dir=True, plot_kwargs=plot_kwargs)
write_all_sequence_audio(audio_path, starts_seq_cut, lengths_seq_cut, timestep, out_dir)



# x Exclusion mask apply
# - Output patterns and audio with plots
# - Store in database
    # - recording_id, seq_num, duration_seq, seq_sec, duration_sec, group number, group rank
# - Quick get next pattern





