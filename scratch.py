
%load_ext autoreload
%autoreload 2

## extract_motives
import os

out_dir = os.path.join("output", 'hpc')
input_filelist = os.path.join(out_dir, "ss_matrices_filelist.txt")
inputs = read_file(input_filelist)

args = {
        'tol': 0.01,
        'rho':2,
        'dom':'audio',
        'ssm_read_pk':False,
        'read_pk':False,
        'n_jobs':10,
        'tonnetz':False,
        'csv_files':'jku_csv_files_test.txt'
    }

if args['csv_files'] is not None:
    csv_files = read_file(args['csv_files'])
else:
    csv_files = None

## extract_motives.process_audio_poly
files=inputs
outdir=out_dir
tol=args['tol']
rho=args['rho']
domain=args['dom']
csv_files=csv_files
ssm_read_pk=args['ssm_read_pk']
read_pk=args['read_pk']
n_jobs=args['n_jobs']
tonnetz=args['tonnetz']

utils.ensure_dir(outdir)
if csv_files is None:
    csv_files = [None] * len(files)

## extract_motives.process_piece
#for wav in files:
wav = files[0]
csv = None



fn_ss_matrix=wav
outdir=outdir
tol=tol
domain=domain
ssm_read_pk=ssm_read_pk
read_pk=read_pk
tonnetz=tonnetz
rho=rho
csv_file=csv

f_base = os.path.basename(fn_ss_matrix)
base_name = os.path.join(outdir, f_base.split(".")[0] + ".seg")

#extractor.process(fn_ss_matrix, base_name, domain, csv_file=csv_file,
#                  tol=tol, ssm_read_pk=ssm_read_pk,
#                  read_pk=read_pk, tonnetz=tonnetz, rho=rho)
wav_file=fn_ss_matrix
outfile=base_name
domain=domain
csv_file=csv_file
tol=tol
bpm=None
ssm_read_pk=ssm_read_pk
read_pk=read_pk
tonnetz=tonnetz
rho=rho
is_ismir=False
sonify=False

from complex_auto.motives_extractor import *
from complex_auto.motives_extractor.extractor import *

min_notes = 16
max_diff_notes = 5

# to process
if wav_file.endswith("wav"):
    # Get the correct bpm if needed
    if bpm is None:
        bpm = get_bpm(wav_file)
    
    h = bpm / 60. / 8.  # Hop size /8 works better than /4, but it takes longer
    # Obtain the Self Similarity Matrix
    X = compute_ssm(wav_file, h, ssm_read_pk, is_ismir, tonnetz)
elif wav_file.endswith("npy"):
    X = np.load(wav_file)
    X = prepro(X)
    if domain == "symbolic":
        h = .25 # 2. # for symbolic (16th notes)
    else:
        if bpm is None:
            bpm = get_bpm(wav_file)
        h = 0.0886 * bpm / 60

offset = 0


######################################################################
## Begin - Plots are only saved for sample sizes sufficiently small ##
######################################################################

# Sample for development
if all([s1,s2]):
    save_imgs = s2-s1 < 4000
    X_samp = X.copy()[s1:s2,s1:s2]
else:
    save_imgs = False
    X_samp = X.copy()

if save_imgs:
    skimage.io.imsave(sim_filename, X_samp)

###################################
## Extract Pich Track From Audio ##
###################################
import librosa

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

import essentia.standard as estd

import pandas as pd 
from scipy.ndimage import gaussian_filter1d

# Save output
def interpolate_below_length(arr, val, gap):
    """
    Interpolate gaps of value, <val> of 
    length equal to or shorter than <gap> in <arr>
    
    :param arr: Array to interpolate
    :type arr: np.array
    :param val: Value expected in gaps to interpolate
    :type val: number
    :param gap: Maximum gap length to interpolate, gaps of <val> longer than <g> will not be interpolated
    :type gap: number

    :return: interpolated array
    :rtype: np.array
    """
    s = np.copy(arr)
    is_zero = s == val
    cumsum = np.cumsum(is_zero).astype('float')
    diff = np.zeros_like(s)
    diff[~is_zero] = np.diff(cumsum[~is_zero], prepend=0)
    for i,d in enumerate(diff):
        if d <= gap:
            s[int(i-d):i] = np.nan
    interp = pd.Series(s).interpolate(method='linear', axis=0)\
                         .ffill()\
                         .bfill()\
                         .values
    return interp


def extract_pitch_track(audio_path, frameSize, hopSize, gap_interp, smooth, sr):

    audio_loaded, _ = librosa.load(audio_path, sr=sr)

    # Run spleeter on track to remove the background
    separator = Separator('spleeter:2stems')
    audio_loader = AudioAdapter.default()
    waveform, _ = audio_loader.load(audio_path, sample_rate=sr)
    prediction = separator.separate(waveform=waveform)
    clean_vocal = prediction['vocals']

    # Prepare audio for pitch extraction
    audio_mono = clean_vocal.sum(axis=1) / 2
    audio_mono_eqloud = estd.EqualLoudness(sampleRate=sr)(audio_mono)

    # Extract pitch using Melodia algorithm from Essentia
    pitch_extractor = estd.PredominantPitchMelodia(frameSize=frameSize, hopSize=hopSize)
    raw_pitch, _ = pitch_extractor(audio_mono_eqloud)
    raw_pitch_ = np.append(raw_pitch, 0.0)
    time = np.linspace(0.0, len(audio_mono_eqloud) / sr, len(raw_pitch))

    timestep = time[4]-time[3] # resolution of time track

    # Gap interpolation
    if gap_interp:
        print(f'Interpolating gaps of {gap_interp} or less')
        raw_pitch = interpolate_below_length(raw_pitch_, 0, int(gap_interp/timestep))
        
    # Gaussian smoothing
    if smooth:
        print(f'Gaussian smoothing with sigma={smooth}')
        pitch = gaussian_filter1d(raw_pitch, smooth)
    else:
        pitch = raw_pitch[:]

    return pitch, raw_pitch, timestep, time


##############
## Binarize ##
##############
def binarize(X, bin_thresh, filename=None):
    X_bin = X.copy()
    X_bin[X_bin < bin_thresh] = 0
    X_bin[X_bin >= bin_thresh] = 1

    if filename:
        skimage.io.imsave(filename, X_bin)

    return X_bin

#######################
## Diagonal Gaussian ##
#######################
from scipy.ndimage import gaussian_filter

def diagonal_gaussian(X, gauss_sigma, filename=False):
    d = X.shape[0]
    X_gauss = X.copy()

    diag_indices_x, diag_indices_y = np.diag_indices_from(X_gauss)
    for i in range(1,d):
        diy = np.append(diag_indices_y, diag_indices_y[:i])
        diy = diy[i:]
        X_gauss[diag_indices_x, diy] = gaussian_filter(X_gauss[diag_indices_x, diy], sigma=gauss_sigma)

    diag_indices_x, diag_indices_y = np.diag_indices_from(X_gauss)
    for i in range(1,d):
        dix = np.append(diag_indices_x, diag_indices_x[:i])
        dix = dix[i:]
        X_gauss[dix, diag_indices_y] = gaussian_filter(X_gauss[dix, diag_indices_y], sigma=gauss_sigma)

    if filename:
        skimage.io.imsave(filename, X_gauss)

    return X_gauss


###########
## Hough ##
###########
import cv2
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm

def plot_hough(image, h, theta, d, peaks, out_file):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]
    ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(image, cmap=cm.gray)
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    for _, angle, dist in zip(*peaks):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.5, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.clf()


def hough_transform(X, min_dist_sec, cqt_window, hough_high_angle, hough_low_angle, hough_threshold, filename=None):
    # TODO: fix this
    hough_min_dist = int(min_dist_sec * cqt_window)

    if hough_high_angle == hough_low_angle:
        tested_angles = np.array([-hough_high_angle * np.pi / 180])
    else:
        tested_angles = np.linspace(- hough_low_angle * np.pi / 180, -hough_high_angle-1 * np.pi / 180, 100, endpoint=False) #np.array([-x*np.pi/180 for x in range(43,47)])

    h, theta, d = hough_line(X, theta=tested_angles)
    peaks = hough_line_peaks(h, theta, d, min_distance=hough_min_dist, min_angle=0, threshold=hough_threshold)
    
    if filename:
        plot_hough(X, h, theta, d, peaks, filename)

    return peaks

#####################
## Path Extraction ##
#####################
def get_extremes(angle, dist, l):
    # Make a line with "num" points...

    # y = (dist - x*np.cos(angle))/np.sin(angle)
    # x = (dist - y*np.sin(angle))/np.cos(angle)

    x0 = 0
    y0 = int((dist - x0*np.cos(angle))/np.sin(angle))

    if (y0 < 0 or y0 > l):
        y0 = 0
        x0 = int((dist - y0*np.sin(angle))/np.cos(angle))

    x1 = l
    y1 = int((dist - x1*np.cos(angle))/np.sin(angle))

    if (y1 < 0 or y1 > l):
        y1 = l
        x1 = int((dist - y1*np.sin(angle))/np.cos(angle))

    return x0, y0, x1, y1


def extract_segments(matrix, angle, dist, min_diff):
    l = matrix.shape[0]-1

    x0, y0, x1, y1 = get_extremes(angle, dist, l)

    # For some reason Hough Transform returns lines 
    # defined outside the grid
    if any([y1>l, x1>l, x0>l, y0>l, y1<0, x1<0, x0<0, y0<0]):
        return []

    length = int(np.hypot(x1-x0, y1-y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
    x = x.astype(int)
    y = y.astype(int)

    # Extract the values along the line
    zi = matrix[x, y]

    # get index of beginning and end of non-zero
    non_zero = np.where(zi != 0)[0]
    if len(non_zero) == 0:
        return []

    segments = []
    this_segment = []
    for i in range(len(non_zero)):
        if i == 0:
            this_segment.append(non_zero[i])
            continue
        if non_zero[i] - non_zero[i-1] <= min_diff:
            continue
        else:
            this_segment.append(non_zero[i-1])
            segments.append(this_segment)
            this_segment = [non_zero[i]]
    this_segment.append(non_zero[-1])
    segments.append(this_segment)

    all_segments = []
    for i1, i2 in segments:
        # (x start, y start), (x end, y end)
        all_segments.append([(x[i1], y[i1]), (x[i2], y[i2])])

    return all_segments


def get_all_segments(X, peaks, min_diff_trav, min_length_cqt):
    all_segments = []
    for _, angle, dist in zip(*peaks):
        segments = extract_segments(X, angle, dist, min_diff_trav)

        # If either of the lengths are above minimum length, add to all segments
        for s in segments:
            x0 = s[0][0]
            y0 = s[0][1]
            x1 = s[1][0]
            y1 = s[1][1]

            l0 = x1-x0
            l1 = y1-y0

            to_add = []
            if max([l1, l0]) > min_length_cqt:
                to_add.append(s)

        all_segments += to_add

    all_segments = sorted([sorted(x) for x in all_segments])

    return all_segments

#   length_change = 1
#   while length_change != 0:
#       l1 = len(all_segments)
#       all_segments = sorted([sorted(x, key=lambda y: (y[0], y[1])) for x in all_segments])
#       all_segments = [all_segments[i] for i in range(len(all_segments)) \
#                           if i == 0 or not \
#                           (same_seqs_marriage(
#                               all_segments[i][0][0], all_segments[i][0][1],
#                               all_segments[i-1][0][0], all_segments[i-1][0][1],
#                                thresh=same_seqs_thresh) and
#                            same_seqs_marriage(
#                               all_segments[i][1][0], all_segments[i][1][1],
#                               all_segments[i-1][1][0], all_segments[i-1][1][1],
#                                thresh=same_seqs_thresh))
#                       ]
#       l2 = len(all_segments)
#       length_change=l2-l1


###########################
## Break Stable Segments ##
###########################
def break_segment(segment_pair, stable_mask, cqt_window, sr, timestep):
    
    # (x start, y start), (x end, y end)
    x_start = segment_pair[0][0]
    x_start_ts = int((x_start*cqt_window)/(sr*timestep))

    x_end = segment_pair[1][0]
    x_end_ts = int((x_end*cqt_window)/(sr*timestep))

    y_start = segment_pair[0][1]
    y_start_ts = int((y_start*cqt_window)/(sr*timestep))

    y_end = segment_pair[1][1]
    y_end_ts = int((y_end*cqt_window)/(sr*timestep))

    stab_x = stable_mask[x_start_ts:x_end_ts]
    stab_y = stable_mask[y_start_ts:y_end_ts]

    # If either sequence contains a stable region, divide
    if any([2 in stab_x, 2 in stab_y]):
        break_points_x = np.where(stab_x==2)[0]
        break_points_y = np.where(stab_y==2)[0]
        if len(break_points_y) > len(break_points_x):
            break_points = break_points_y
        else:
            break_points = break_points_x
    else:
        return []

    break_points_x = [x_start + int(x*(sr*timestep)/cqt_window) for x in break_points]
    break_points_y = [y_start + int(x*(sr*timestep)/cqt_window) for x in break_points]

    new_segments = []
    for i in range(len(break_points_x)):
        bx = break_points_x[i]
        by = break_points_y[i]

        if i == 0:
            new_segments.append([(x_start, y_start), (bx, by)])
        else:
            bx1 = break_points_x[i-1]
            by1 = break_points_y[i-1]

            new_segments.append([(bx1, by1), (bx, by)])

    new_segments.append([(bx, by), (x_end, y_end)])

    return new_segments


def break_all_segments(all_segments, stable_mask, cqt_window, sr, timestep):
    all_broken_segments = []
    for segment_pair in all_segments:
        broken = break_segment(segment_pair, stable_mask, cqt_window, sr, timestep)
        if len(broken) > 0:
            all_broken_segments += broken
        else:
            all_broken_segments += [segment_pair]

    return sorted([sorted(x) for x in all_broken_segments])


####################
## Group segments ##
####################
from sklearn.cluster import DBSCAN
from itertools import groupby
from operator import itemgetter

def same_seqs_marriage(x1, y1, x2, y2, thresh=4):
    return (abs(x1-x2) < thresh) and (abs(y1-y2) < thresh)


def get_length(x1,y1,x2,y2):
    return ((x2-x1)**2 + (y2-y1)**2)**0.5


def remove_group_duplicates(group, eps):
    start_length = sorted([(x1, x2-x1, i) for i, (x1,x2) in enumerate(group) if x2-x1>0])

    clustering = DBSCAN(eps=eps, min_samples=1)\
                    .fit(np.array([d for d,l,i in start_length])\
                    .reshape(-1, 1))
    
    with_cluster = list(zip(clustering.labels_, start_length))

    top_indices = []
    for g, data in groupby(with_cluster, key=itemgetter(0)):
        data = list(data)
        top_i = sorted(data, key=lambda y: -y[1][1])[0][1][2]
        top_indices.append(top_i)

    group_reduced = [x for i,x in enumerate(group) if i in top_indices]

    return group_reduced

def group_segments(all_segments, same_seqs_thresh):
    all_seg_copy = all_segments.copy()
    all_groups = []
    skip_array = [0]*len(all_seg_copy)
    for i in range(len(all_seg_copy)):
        # If this segment has been grouped already, do not consider
        if skip_array[i] == 1:
            continue
        
        # To avoid altering original segments array
        this_seg = all_seg_copy[i]

        x0 = this_seg[0][0]
        y0 = this_seg[0][1]
        x1 = this_seg[1][0]
        y1 = this_seg[1][1]

        this_group = [(x0,x1),(y0,y1)]

        for i_rest, seg in enumerate(all_seg_copy[i+1:], i+1):
            x_0 = seg[0][0]
            y_0 = seg[0][1]
            x_1 = seg[1][0]
            y_1 = seg[1][1]

            x_match = same_seqs_marriage(x0, x1, x_0, x_1, thresh=same_seqs_thresh)
            y_match = same_seqs_marriage(y0, y1, y_0, y_1, thresh=same_seqs_thresh)

            if x_match and y_match:
                # same pattern, do not add
                skip_array[i_rest] = 1
                continue

            if x_match and not y_match:
                # dont append the one that matches (since we already have it)
                this_group += [(y_0, y_1)]
                skip_array[i_rest] = 1
            
            elif y_match and not x_match:
                # dont append the one that matches (since we already have it)
                this_group += [(x_0, x_1)]
                skip_array[i_rest] = 1

        all_groups.append(this_group)

    return [remove_group_duplicates(g, eps=same_seqs_thresh) for g in all_groups]


######################################
## Convert to Pitch Track Timesteps ##
######################################
def convert_seqs_to_timestep(all_groups, cqt_window, sr, timestep):
    lengths = []
    starts = []
    for group in all_groups:
        this_l = []
        this_s = []
        for g in group:
            l = g[1]-g[0]
            s = g[0]
            if l > 0:
                this_l.append(l)
                this_s.append(s)
        lengths.append(this_l)
        starts.append(this_s)

    starts_sec = [[x*cqt_window/sr for x in p] for p in starts]
    lengths_sec = [[x*cqt_window/sr for x in l] for l in lengths]

    starts_seq = [[int(x/timestep) for x in p] for p in starts_sec]
    lengths_seq = [[int(x/timestep) for x in l] for l in lengths_sec]

    return starts_seq, lengths_seq


###########################
# Reduce/Apply Exclusions #
###########################
from exploration.sequence import contains_silence, min_gap, too_stable

exclusion_functions = [contains_silence, min_gap, too_stable]

def apply_exclusions(starts_seq, lengths_seq, exclusion_functions):
    for i in range(len(starts_seq)):
        these_seq = starts_seq[i]
        these_lens = lengths_seq[i]
        for j in range(len(these_seq))[::-1]:
            this_len = these_lens[j]
            this_start = these_seq[j]
            n_fails = 0
            for func in exclusion_functions:
                if func(raw_pitch[this_start:this_start+this_len]):
                    n_fails += 1
            if n_fails > 0:
                these_seq.pop(j)
                these_lens.pop(j)

    # minimum number in group to be pattern group
    starts_seq_exc = [seqs for seqs in starts_seq if len(seqs)>=min_in_group]
    lengths_seq_exc = [seqs for seqs in lengths_seq if len(seqs)>=min_in_group]

    return starts_seq_exc, lengths_seq_exc

# remove duplicates
#   same_seqs_thresh_seq = int(same_seqs_thresh_secs/timestep) # max distance to group measured in pitch track timesteps

#   starts_seq_exc = []
#   lengths_seq_exc = []
#   for i in range(len(starts_seq)):
#       this_group = sorted(list(zip(starts_seq[i], lengths_seq[i])))
#       starts_red = []
#       len_red = []
#       for j in range(len(this_group)):

#           # this pattern details
#           this_start = this_group[j][0]
#           this_length = this_group[j][1]

#           # if first pattern, include it
#           if j == 0:
#               starts_red.append(this_start)
#               len_red.append(this_length)
#               continue

#           # previous pattern details
#           that_start = this_group[j-1][0]
#           that_length = this_group[j-1][1]

#           # is this pattern and the previous sufficiently similar?
#           if_group = same_seqs_marriage(
#               this_start, this_length, that_start, 
#               that_length, thresh=same_seqs_thresh_seq)

#           # If not, add this pattern to group
#           if not if_group:
#               starts_red.append(this_start)
#               len_red.append(this_length)

#       # add reduced patterns to new list        
#       starts_seq_exc.append(starts_red)
#       lengths_seq_exc.append(len_red)

################
## Evaluation ##
################
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

annotations_path = '../carnatic-motifs/Akkarai_Sisters_-_Koti_Janmani_multitrack-vocal_-_ritigowla.TextGrid'

annotations_orig = load_annotations(annotations_path)
annotations = annotations_orig.copy()

def is_match(sp, lp, sa, ea, tol=0.1):

    ep = sp + lp

    if (sa - tol <= sp <= sa + tol) and (ea - tol <= ep <= ea + tol):
        return 'full match'

    elif ((sp > sa) and (sp <= ea)) or ((ep > sa) and (ep <= ea)):
        return 'partial match'

    else:
        return None


def evaluate_annotations(annotations, starts_sec_exc, lengths_sec_exc, tol):

    annotations = annotations.copy()
    results_dict = {}
    group_num_dict = {}
    occ_num_dict = {}
    for i, seq_group in enumerate(starts_sec_exc):
        length_group  = lengths_sec_exc[i]
        for j, seq in enumerate(seq_group):
            length = length_group[j]
            red_annot = annotations[
                            (annotations['s1'] >= seq - tol) &
                            (annotations['s2'] <= seq + length + tol)
                        ]
            for ai, (tier, s1, s2, text) in zip(red_annot.index, red_annot.values):
                matched = is_match(seq, length, s1, s2, tol=tol)
                if matched:
                    results_dict[ai] = matched
                    group_num_dict[ai] = i
                    occ_num_dict[ai] = j


    annotations['match'] = ['no match' if not x in results_dict else results_dict[x] for x in annotations.index]
    annotations['group_num'] = [None if not x in group_num_dict else group_num_dict[x] for x in annotations.index]
    annotations['occ_num'] = [None if not x in occ_num_dict else occ_num_dict[x] for x in annotations.index]

    return annotations


def get_metrics(annotations, starts_sec_exc):
    
    if not starts_sec_exc:
        return {
            'n_groups': 0,
            'n_patterns': 0,
            'max_n_in_group': np.nan,
            'partial_match_precision': np.nan,
            'partial_match_recall': np.nan,
            'full_match_precision': np.nan,
            'full_match_recall': np.nan,
            'partial_group_precision': np.nan,
            'full_group_precision': np.nan,
        }
    n_patterns = sum([len(x) for x in starts_sec_exc])
    n_groups = len(starts_sec_exc)
    max_n_in_group = max([len(x) for x in starts_sec_exc])

    partial_match_df = annotations[annotations['match'] != 'no match']
    full_match_df = annotations[annotations['match'] == 'full match']

    # partial match
    partial_prec = len(partial_match_df)/n_patterns
    partial_rec = len(partial_match_df)/len(annotations)

    # full match
    full_prec = len(full_match_df)/n_patterns
    full_rec = len(full_match_df)/len(annotations)

    # groups with a partial match
    group_partial_prec = partial_match_df['group_num'].nunique()/n_groups

    # groups with a full match
    group_full_prec = full_match_df['group_num'].nunique()/n_groups

    return {
        'n_groups': n_groups,
        'n_patterns': n_patterns,
        'max_n_in_group': max_n_in_group,
        'partial_match_precision': partial_prec,
        'partial_match_recall': partial_rec,
        'full_match_precision': full_prec,
        'full_match_recall': full_rec,
        'partial_group_precision': group_partial_prec,
        'full_group_precision': group_full_prec,
    }



###############################
## Edge Detection Parameters ##
###############################

import skimage.io
import skimage.feature
import sys

# Output paths of each step in pipeline
out_dir = os.path.join("output", 'all_vocal')

sim_filename = os.path.join(out_dir, 'Koti Janmani_simsave.png')
gauss_filename = os.path.join(out_dir, 'Koti Janmani_gauss.png')
edge_filename = os.path.join(out_dir, 'Koti Janmani_edges.png')
bin_filename = os.path.join(out_dir, 'Koti Janmani_binary.png')
cont_filename = os.path.join(out_dir, 'Koti Janmani_cont.png')
hough_filename = os.path.join(out_dir, 'Koti Janmani_hough.png')

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
audio_path = 'audio/Akkarai Sisters at Arkay by Akkarai Sisters/Koti Janmani/Koti Janmani.multitrack-vocal.mp3'

# stability identification
stab_hop_secs = 0.2 # window size for stab computations in seconds
min_stability_length_secs = 1.0 # minimum legnth of region to be considered stable in seconds
freq_var_thresh_stab = 10 # max variation in pitch to be considered stable region

# Binarize raw sim array 0/1 below and above this value...
bin_thresh = 0.1

# Gaussian filter along diagonals with sigma...
gauss_sigma = 18

# After gaussian, re-binarize with this threshold
cont_thresh = 0.15

# Hough transform parameters
min_dist_sec = 0 # min dist in seconds between lines
hough_threshold = 75

# Only search for lines between these angles (45 corresponds to main diagonal)
hough_low_angle = 45.5
hough_high_angle = 44.5

# distance between consecutive diagonals to be joined
min_diff_trav = 25

# Grouping diagonals
min_pattern_length_seconds = 2
min_length_cqt = min_pattern_length_seconds*sr/cqt_window
min_in_group = 3 # minimum number of patterns to be included in pattern group

# Minimum distance between start points (x,y) to be joined a segment group
same_seqs_thresh_secs = 1
same_seqs_thresh = int(same_seqs_thresh_secs*sr/cqt_window)

eval_tol = 0.2


##############
## Pipeline ##
##############
from exploration.sequence import get_stability_mask, add_center_to_mask

print('Extracting pitch track')
pitch, raw_pitch, timestep, time = extract_pitch_track(audio_path, frameSize, hopSize, gap_interp, smooth, sr)

print('Computing stability mask')
stable_mask = get_stability_mask(raw_pitch, min_stability_length_secs, stab_hop_secs, freq_var_thresh_stab, timestep)
silence_mask = raw_pitch == 0
silence_mask = add_center_to_mask(silence_mask)

print('Binarizing similarity matrix')
X_bin = binarize(X, bin_thresh)

print('Applying diagonal gaussian filter')
X_gauss = diagonal_gaussian(X_bin, gauss_sigma)

print('Binarize gaussian blurred similarity matrix')
X_cont = binarize(X_gauss, cont_thresh)

print('Applying Hough Transform')
peaks = hough_transform(
    X_cont, min_dist_sec, cqt_window, 
    hough_high_angle, hough_low_angle, hough_threshold)

print('Extracting paths')
all_segments = get_all_segments(X_cont, peaks, min_diff_trav, min_length_cqt)

print('Breaking segments with stable regions')
all_broken_segments = break_all_segments(all_segments, stable_mask, cqt_window, sr, timestep)

print('Breaking segments with silent regions')
all_broken_segments = break_all_segments(all_broken_segments, silence_mask, cqt_window, sr, timestep)

print('Grouping Segments')
all_groups = group_segments(all_broken_segments, same_seqs_thresh)

print('Convert sequences to pitch track timesteps')
starts_seq, lengths_seq = convert_seqs_to_timestep(all_groups, cqt_window, sr, timestep)

print('Applying exclusion functions')
starts_seq_exc, lengths_seq_exc = apply_exclusions(starts_seq, lengths_seq, exclusion_functions)

starts_sec_exc = [[x*timestep for x in p] for p in starts_seq_exc]
lengths_sec_exc = [[x*timestep for x in l] for l in lengths_seq_exc]

print('Evaluating')
annotations_orig = load_annotations(annotations_path)
annotations = evaluate_annotations(annotations_orig, starts_sec_exc, lengths_sec_exc, eval_tol)

metrics = get_metrics(annotations, starts_sec_exc)


############
## Output ##
############

from exploration.visualisation import plot_all_sequences, plot_pitch
from exploration.io import write_all_sequence_audio, load_yaml
from exploration.pitch import cents_to_pitch, pitch_seq_to_cents, pitch_to_cents

svara_cent_path = "conf/svara_cents.yaml"
svara_freq_path = "conf/svara_lookup.yaml"

svara_cent = load_yaml(svara_cent_path)
svara_freq = load_yaml(svara_freq_path)

tonic = 195.99

yticks_dict = {k:cents_to_pitch(v, tonic) for k,v in svara_cent.items()}
yticks_dict = {k:v for k,v in yticks_dict.items() if any([x in k for x in ['S', 'R2', 'G2', 'M1', 'P', 'D2', 'N2', 'S']])}

plot_kwargs = {
    'yticks_dict':yticks_dict,
    'cents':True,
    'tonic':195.997718,
    'emphasize':['S', 'S^'],
    'figsize':(15,4)
}

top_n = 1000
plot_all_sequences(raw_pitch, time, lengths_seq_exc[:top_n], starts_seq_exc[:top_n], 'output/test', clear_dir=True, plot_kwargs=plot_kwargs)
write_all_sequence_audio(audio_path, starts_seq_exc[:top_n], lengths_seq_exc[:top_n], timestep, 'output/test')







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





