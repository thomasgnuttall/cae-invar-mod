#%load_ext autoreload
#%autoreload 2
import sys
import datetime
import os
import pickle

import skimage.io

from complex_auto.motives_extractor import *
from complex_auto.motives_extractor.extractor import *

from exploration.pitch import extract_pitch_track
from exploration.img import (
    remove_diagonal, convolve_array_tile, binarize, diagonal_gaussian, 
    hough_transform, hough_transform_new, scharr, sobel,
    apply_bin_op, make_symmetric, edges_to_contours)
from exploration.segments import (
    extract_segments_new, get_all_segments, break_all_segments, do_patterns_overlap, reduce_duplicates, 
    remove_short, extend_segments, join_all_segments, extend_groups_to_mask, group_segments, group_overlapping,
    group_by_distance)
from exploration.sequence import (
    apply_exclusions, contains_silence, min_gap, too_stable, 
    convert_seqs_to_timestep, get_stability_mask, add_center_to_mask,
    remove_below_length, extend_to_mask)
from exploration.evaluation import evaluate, load_annotations_brindha, get_coverage
from exploration.visualisation import plot_all_sequences, plot_pitch, flush_matplotlib
from exploration.io import load_sim_matrix, write_all_sequence_audio, load_yaml, load_pkl, create_if_not_exists, write_pkl, run_or_cache
from exploration.pitch import cents_to_pitch, pitch_seq_to_cents, pitch_to_cents, get_timeseries, interpolate_below_length

################
## Parameters ##
################
track_name = 'Koti Janmani'

# Sample rate of audio
sr = 44100

# size in frames of cqt window from convolution model
cqt_window = 1984 # was previously set to 1988

# Take sample of data, set to None to use all data
s1 = 4000 # lower bound index (5000 has been used for testing)
s2 = 8001 # higher bound index (9000 has been used for testing)

# pitch track extraction
gap_interp = 250*0.001 # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]

# stability identification
stab_hop_secs = 0.2 # window size for stab computations in seconds
min_stability_length_secs = 1.0 # minimum length of region to be considered stable in seconds
freq_var_thresh_stab = 10 # max variation in pitch to be considered stable region

conv_filter_str = 'sobel'

# Binarize raw sim array 0/1 below and above this value...
# depends completely on filter passed to convolutional step
# Best...
#   scharr, 0.56
#   sobel unidrectional, 0.1
#   sobel bidirectional, 0.15
bin_thresh = np.arange(0.08, 0.26, 0.02)
# lower bin_thresh for areas surrounding segments
bin_thresh_segment = np.arange(0.04, 0.14, 0.02)

# percentage either size of a segment considered for lower bin thresh
perc_tail = 0.5

# Gaussian filter along diagonals with sigma...
gauss_sigma = None

# After gaussian, re-binarize with this threshold
cont_thresh = 0.15

# morphology params
etc_kernel_size = 10 # For closing
binop_dim = 3 # square dimension of binary opening structure (square matrix of zeros with 1 across the diagonal)

# Distance between consecutive diagonals to be joined in seconds
min_diff_trav = 0.3

# Grouping diagonals
min_pattern_length_seconds = 2
min_in_group = 2 # minimum number of patterns to be included in pattern group

# Joining groups
match_tol = 3


n_dtw = 5 # number of samples to take from each group to compare dtw values
thresh_dtw = None # if average dtw distance (per sequence) between two groups is below this threshold, group them
thresh_cos = None  # if average cos distance (per sequence) between two groups is below this threshold, group them
# Two segments must overlap in both x and y by <dupl_perc_overlap> 
# to be considered the same, only the longest is returned
dupl_perc_overlap = 0.95

# extend to silent/stability mask using this proportion of pattern
ext_mask_tol = 0.33

# Exclusions
exclusion_functions = [contains_silence]

# evaluation
annotations_path = 'annotations/koti_janmani.txt'
partial_perc = 0.66 # how much overlap does an annotated and identified pattern needed to be considered a partial match

# limit the number of groups outputted
top_n = 1000

write_plots = False
write_audio = False
write_patterns = False
write_annotations = False

from scratch import main

all_bt = []
all_bts = []
all_recall = []
all_precision = []
all_f1 = []
for bin_thresh in np.arange(0.08, 0.18, 0.02):
    for bin_thresh_segment in np.arange(0.03, 0.13, 0.02):
        if bin_thresh > bin_thresh_segment:
            recall, precision, f1 = main(
                    track_name, 'gridsearch', sr, cqt_window, s1, s2,
                    gap_interp, stab_hop_secs, min_stability_length_secs, 
                    freq_var_thresh_stab, conv_filter_str, bin_thresh,
                    bin_thresh_segment, perc_tail,  gauss_sigma, cont_thresh,
                    etc_kernel_size, binop_dim, min_diff_trav, min_pattern_length_seconds,
                    min_in_group, match_tol, ext_mask_tol, n_dtw, thresh_dtw, thresh_cos,
                    dupl_perc_overlap, exclusion_functions, annotations_path, partial_perc, 
                    top_n, write_plots, write_audio, write_patterns, write_annotations)
            all_bt.append(bin_thresh)
            all_bts.append(bin_thresh_segment)
            all_recall.append(recall)
            all_precision.append(precision)
            all_f1.append(f1)

write_pkl(list(zip(all_bt, all_bts, all_recall, all_precision, all_f1)), 'GRIDSEARCH_RESULTS.pkl')
