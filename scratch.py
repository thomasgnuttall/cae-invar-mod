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

### What do we have?
# Track Names...
#   all_tracks = [
#       'Koti Janmani', 
#       'Karunimpa Idi',
#       'Sharanu Janakana',
#       'Siddhi Vinayakam',
#       'Vanajaksha Ninne Kori',
#       'Sundari Nee Divya',
#       'Shankari Shankuru',
#       'Karuna Nidhi Illalo', 
#       'Koluvamaregatha', 
#       'Mati Matiki', 
#       'Ramabhi Rama Manasu',
#       'Palisomma Muddu Sarade',
#       'Rama Rama Guna Seema',
#       'Lokavana Chatura']
all_tracks = [
    'Koti Janmani', 
    'Vanajaksha Ninne Kori',
    'Sharanu Janakana',
    'Sundari Nee Divya'
]
#for t in all_tracks:
#    out_dir = os.path.join(f'/Volumes/Shruti/asplab2/cae-invar/data/self_similarity/{t}/')
#    metadata_path = os.path.join(out_dir, 'metadata.pkl')
#    metadata = load_pkl(metadata_path)
#    title = t
#    raga = metadata['raga']
#    line = '-'*len(title)
#    ass = metadata['sparse_size']
#    print(f'{title}')
#    if title:
#        print(f'{line}')
#    print(f'    Raga: {raga}')
#    print(f'    Sparse array size: ({ass}, {ass})')

################
## Parameters ##
################
#track_name = 'Sharanu Janakana'
track_name = 'Vanajaksha Ninne Kori'

# Sample rate of audio
sr = 44100

# size in frames of cqt window from convolution model
cqt_window = 1984 # was previously set to 1988

# Take sample of data, set to None to use all data
s1 = 4000 # lower bound index (5000 has been used for testing)
s2 = 8000 # higher bound index (9000 has been used for testing)

# pitch track extraction
gap_interp = 250*0.001 # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]

# stability identification
stab_hop_secs = 0.2 # window size for stab computations in seconds
min_stability_length_secs = 1.0 # minimum legnth of region to be considered stable in seconds
freq_var_thresh_stab = 10 # max variation in pitch to be considered stable region

conv_filter_str = 'sobel'

# Binarize raw sim array 0/1 below and above this value...
# depends completely on filter passed to convolutional step
# Best...
#   scharr, 0.56
#   sobel unidrectional, 0.1
#   sobel bidirectional, 0.15
bin_thresh = 0.20
# lower bin_thresh for areas surrounding segments
bin_thresh_segment = 0.13
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
match_tol = 1
# extend to silent/stability mask using this proportion of pattern
ext_mask_tol = 0.5

n_dtw = 5 # number of samples to take from each group to compare dtw values
thresh_dtw = None # if average dtw distance (per sequence) between two groups is below this threshold, group them
thresh_cos = None  # if average cos distance (per sequence) between two groups is below this threshold, group them
# Two segments must overlap in both x and y by <dupl_perc_overlap> 
# to be considered the same, only the longest is returned
dupl_perc_overlap = 0.95

# Exclusions
exclusion_functions = [contains_silence]

# evaluation
#annotations_path = 'annotations/koti_janmani.txt'
annotations_path = 'annotations/vanajaksha_ninne_kori.txt'

partial_perc = 0.66 # how much overlap does an annotated and identified pattern needed to be considered a partial match

# limit the number of groups outputted
top_n = 1000

write_plots = True
write_audio = True
write_patterns = True
write_annotations = True

def main(
    track_name, run_name, sr, cqt_window, s1, s2,
    gap_interp, stab_hop_secs, min_stability_length_secs, 
    freq_var_thresh_stab, conv_filter_str, bin_thresh,
    bin_thresh_segment, perc_tail,  gauss_sigma, cont_thresh,
    etc_kernel_size, binop_dim, min_diff_trav, min_pattern_length_seconds,
    min_in_group, match_tol, ext_mask_tol, n_dtw, thresh_dtw, thresh_cos,
    dupl_perc_overlap, exclusion_functions,
    annotations_path, partial_perc, top_n, write_plots, 
    write_audio, write_patterns, write_annotations):
    if write_annotations and not annotations_path:
        print('WARNING: write_annotations==True but no annotations path has been passed, annotations will not be written')
    ## Get Data
    out_dir = os.path.join(f'/Volumes/Shruti/asplab2/cae-invar/data/self_similarity/{track_name}/')
    metadata_path = os.path.join(out_dir, 'metadata.pkl')
    sim_path = os.path.join(out_dir, 'self_sim.npy')
    
    print(f'Loading metadata from {metadata_path}')
    metadata = load_pkl(metadata_path)
    arr_sparse_size = metadata['sparse_size']
    raga = metadata['raga']
    print(f'Raga: {raga}')
    print(f'Sparse array size: {arr_sparse_size}')

    print(f'Loading self similarity from {sim_path}')
    X = load_sim_matrix(sim_path)

    ## Unpack Metadata
    arr_orig_size = metadata['orig_size']
    arr_sparse_size = metadata['sparse_size']
    orig_sparse_lookup = metadata['orig_sparse_lookup']
    sparse_orig_lookup = {v:k for k,v in orig_sparse_lookup.items()}
    sparse_orig_lookup[X.shape[0]] = sparse_orig_lookup[X.shape[0]-1]
    boundaries_orig = metadata['boundaries_orig']
    boundaries_sparse = metadata['boundaries_sparse']
    audio_path = metadata['audio_path']
    pitch_path = metadata['pitch_path']
    stability_path = metadata['stability_path']
    raga = metadata['raga']
    tonic = metadata['tonic']

    print('Loading pitch track')
    raw_pitch, time, timestep = get_timeseries(pitch_path)
    raw_pitch[np.where(raw_pitch<80)[0]]=0
    raw_pitch = interpolate_below_length(raw_pitch, 0, (gap_interp/timestep))

    print('Computing stability/silence mask')
    stable_mask = get_stability_mask(raw_pitch, min_stability_length_secs, stab_hop_secs, freq_var_thresh_stab, timestep)
    silence_mask = (raw_pitch == 0).astype(int)
    silence_mask = add_center_to_mask(silence_mask)
    silence_and_stable_mask = np.array([int(any([i,j])) for i,j in zip(silence_mask, stable_mask)])

    ### Image Processing
    # convolutional filter
    if conv_filter_str == 'sobel':
        conv_filter = sobel

    min_diff_trav_hyp = (2*min_diff_trav**2)**0.5 # translate min_diff_trav to correpsonding diagonal distance
    min_diff_trav_seq = min_diff_trav_hyp*sr/cqt_window

    min_length_cqt = min_pattern_length_seconds*sr/cqt_window

    # Output
    svara_cent_path = "conf/svara_cents.yaml"
    svara_freq_path = "conf/svara_lookup.yaml"

    svara_cent = load_yaml(svara_cent_path)
    svara_freq = load_yaml(svara_freq_path)
    
    if raga in svara_freq:
        arohana = svara_freq[raga]['arohana']
        avorahana = svara_freq[raga]['avorahana']
        all_svaras = list(set(arohana+avorahana))
        print(f'Svaras for raga, {raga}:')
        print(f'   arohana: {arohana}')
        print(f'   avorahana: {avorahana}')

        yticks_dict = {k:cents_to_pitch(v, tonic) for k,v in svara_cent.items()}
        yticks_dict = {k:v for k,v in yticks_dict.items() if any([x in k for x in all_svaras])}

        plot_kwargs = {
            'yticks_dict':yticks_dict,
            'cents':True,
            'tonic':tonic,
            'emphasize':['S', 'S ', 'S  ', ' S', '  S'],
            'figsize':(15,4)
        }
    else:
        plot_kwargs = {
            'yticks_dict':{},
            'cents':False,
            'tonic':None,
            'emphasize':[],
            'figsize':(15,4)
        }
        print(f'No svara information for raga, {raga}')

    ####################
    ## Load sim array ##
    ####################
    # Sample for development
    if all([s1,s2]):
        save_imgs = s2-s1 <= 4000
        X_samp = X.copy()[s1:s2,s1:s2]
    else:
        save_imgs = False
        X_samp = X.copy()

    sim_filename = os.path.join(out_dir, 'progress_plots', '1_simsave.png') if save_imgs else None
    conv_filename = os.path.join(out_dir, 'progress_plots', '2_conv.png') if save_imgs else None
    bin_filename = os.path.join(out_dir, 'progress_plots', '3_binary.png') if save_imgs else None
    diag_filename = os.path.join(out_dir, 'progress_plots', '4_diag.png') if save_imgs else None
    gauss_filename = os.path.join(out_dir, 'progress_plots', '5_gauss.png') if save_imgs else None
    cont_filename = os.path.join(out_dir, 'progress_plots', '6_cont.png') if save_imgs else None
    binop_filename = os.path.join(out_dir, 'progress_plots', '7_binop.png') if save_imgs else None
    hough_filename = os.path.join(out_dir, 'progress_plots', '8_hough.png') if save_imgs else None
    ext_filename = os.path.join(out_dir, 'progress_plots', '9_cont_ext.png') if save_imgs else None


    if save_imgs:
        create_if_not_exists(sim_filename)
        skimage.io.imsave(sim_filename, X_samp)

    ##############
    ## Pipeline ##
    ##############
    print('Convolving similarity matrix')
    X_conv = convolve_array_tile(X_samp, cfilter=conv_filter)

    if save_imgs:
        skimage.io.imsave(conv_filename, X_conv)

    print('Binarizing convolved array')
    X_bin = binarize(X_conv, bin_thresh, filename=bin_filename)
    #X_bin = binarize(X_conv, 0.05, filename=bin_filename)

    print('Removing diagonal')
    X_diag = remove_diagonal(X_bin)

    if save_imgs:
        skimage.io.imsave(diag_filename, X_diag)

    if gauss_sigma:
        print('Applying diagonal gaussian filter')
        diagonal_gaussian(X_bin, gauss_sigma, filename=gauss_filename)

        print('Binarize gaussian blurred similarity matrix')
        binarize(X_gauss, cont_thresh, filename=cont_filename)
    else:
        X_gauss = X_diag
        X_cont  = X_gauss

    print('Ensuring symmetry between upper and lower triangle in array')
    X_sym = make_symmetric(X_cont)

    print('Identifying and isolating regions between edges')
    X_fill = edges_to_contours(X_sym, etc_kernel_size)

    print('Cleaning isolated non-directional regions using morphological opening')
    X_binop = apply_bin_op(X_fill, binop_dim)

    print('Ensuring symmetry between upper and lower triangle in array')
    X_binop = make_symmetric(X_binop)

    if save_imgs:
        skimage.io.imsave(binop_filename, X_binop)
    
    # Hash all parameters used before segment finding to hash results later
    seg_hash = str((
        s1, s2, conv_filter_str, bin_thresh, gauss_sigma,
        cont_thresh, etc_kernel_size, binop_dim
    ))

    segment_path = os.path.join(out_dir, f'segments/{seg_hash}.pkl')
    all_segments = run_or_cache(extract_segments_new, [X_binop], segment_path)

    print('Extending Segments')  
    # Hash all parameters used before segment finding to hash results later
    seg_hash = str((
        s1, s2, conv_filter_str, bin_thresh, gauss_sigma,
        cont_thresh, etc_kernel_size, binop_dim, perc_tail, bin_thresh_segment
    ))

    segment_path = os.path.join(out_dir, f'segments_extended/{seg_hash}.pkl')
    all_segments_extended = run_or_cache(extend_segments, [all_segments, X_sym, X_conv, perc_tail, bin_thresh_segment], segment_path)

    print(f'    {len(all_segments_extended)} extended segments...')

    from exploration.segments import line_through_points

    print('Converting sparse segment indices to original')
    boundaries_sparse = [x for x in boundaries_sparse if x != 0]
    all_segments_scaled_x = []
    for seg in all_segments_extended:
        ((x0, y0), (x1, y1)) = seg
        get_x, get_y = line_through_points(x0, y0, x1, y1)
        
        boundaries_in_x = sorted([i for i in boundaries_sparse if i >= x0 and i <= x1])
        current_x0 = x0
        if boundaries_in_x:
            for b in boundaries_in_x:
                x0_ = current_x0
                x1_ = b
                y0_ = int(get_y(x0_))
                y1_ = int(get_y(x1_))
                all_segments_scaled_x.append(((x0_, y0_), (x1_, y1_)))
                current_x0 = b

            x0_ = current_x0
            x1_ = x1
            y0_ = int(get_y(x0_))
            y1_ = int(get_y(x1_))
            all_segments_scaled_x.append(((x0_, y0_), (x1_, y1_)))
        else:
            all_segments_scaled_x.append(((x0, y0), (x1, y1)))

    all_segments_scaled = []
    for seg in all_segments_scaled_x:
        ((x0, y0), (x1, y1)) = seg
        get_x, get_y = line_through_points(x0, y0, x1, y1)
        
        boundaries_in_y = sorted([i for i in boundaries_sparse if i >= y0 and i <= y1])
        current_y0 = y0
        if boundaries_in_y:
            for b in boundaries_in_y:
                y0_ = current_y0
                y1_ = b
                x0_ = int(get_x(y0_))
                x1_ = int(get_x(y1_))
                all_segments_scaled.append(((x0_, y0_), (x1_, y1_)))
                current_y0 = b
            
            y0_ = current_y0
            y1_ = y1
            x0_ = int(get_x(y0_))
            x1_ = int(get_x(y1_))
            all_segments_scaled.append(((x0_, y0_), (x1_, y1_)))
        else:
            all_segments_scaled.append(((x0, y0), (x1, y1)))

    all_segments_converted = []
    for seg in all_segments_scaled:
        ((x0, y0), (x1, y1)) = seg
        x0_ = sparse_orig_lookup[x0]
        y0_ = sparse_orig_lookup[y0]
        x1_ = sparse_orig_lookup[x1]
        y1_ = sparse_orig_lookup[y1]
        all_segments_converted.append(((x0_, y0_), (x1_, y1_)))

    print('Joining segments that are sufficiently close')
    # Hash all parameters used before segment finding to hash results later
    seg_hash = str((
        s1, s2, conv_filter_str, bin_thresh, gauss_sigma,
        cont_thresh, etc_kernel_size, binop_dim, perc_tail, 
        bin_thresh_segment, min_diff_trav_seq
    ))

    segment_path = os.path.join(out_dir, f'segments_joined/{seg_hash}.pkl')
    all_segments_joined = run_or_cache(join_all_segments, [all_segments_converted, min_diff_trav_seq], segment_path)
    print(f'    {len(all_segments_joined)} joined segments...')

    print('Breaking segments with silent/stable regions')
    # Format - [[(x,y), (x1,y1)],...]
    all_broken_segments = break_all_segments(all_segments_joined, silence_mask, cqt_window, sr, timestep)
    all_broken_segments = break_all_segments(all_broken_segments, stable_mask, cqt_window, sr, timestep)
    print(f'    {len(all_broken_segments)} broken segments...')

    #[(i,((x0,y0), (x1,y1))) for i,((x0,y0), (x1,y1)) in enumerate(all_segments) if x1-x0>10000]
    print('Reducing Segments')
    all_segments_reduced = remove_short(all_broken_segments, min_length_cqt)
    print(f'    {len(all_segments_reduced)} segments above minimum length of {min_pattern_length_seconds}s...')

    print('Identifying Segment Groups')
    all_groups = group_segments(all_segments_reduced, min_length_cqt, match_tol, silence_and_stable_mask, cqt_window, timestep, sr)
    print(f'    {len(all_groups)} segment groups found...')

    print('Extending segments to silence')
    silence_and_stable_mask_2 = np.array([1 if any([i==2, j==2]) else 0 for i,j in zip(silence_mask, stable_mask)])
    all_groups_ext = extend_groups_to_mask(all_groups, silence_and_stable_mask_2, cqt_window, sr, timestep, toler=ext_mask_tol)

    print('Joining Groups of overlapping Sequences')
    all_groups_over = group_overlapping(all_groups_ext, dupl_perc_overlap)
    print(f'    {len(all_groups_over)} groups after join...')

    if thresh_dtw and thresh_cos:
        print('Joining geometrically close groups using pitch tracks')
        all_groups_dtw = group_by_distance(all_groups_over, raw_pitch, n_dtw, thresh_dtw, thresh_cos, cqt_window, sr, timestep)
        print(f'    {len(all_groups_dtw)} groups after join...')
    else:
        all_groups_dtw = all_groups_over

    print('Convert sequences to pitch track timesteps')
    starts_seq, lengths_seq = convert_seqs_to_timestep(all_groups_dtw, cqt_window, sr, timestep)

    print('Applying exclusion functions')
    starts_seq_exc,  lengths_seq_exc = remove_below_length(starts_seq, lengths_seq, timestep, min_pattern_length_seconds)

    starts = [p for p in starts_seq_exc if len(p)>=min_in_group]
    lengths = [p for p in lengths_seq_exc if len(p)>=min_in_group]

    starts_sec = [[x*timestep for x in p] for p in starts]
    lengths_sec = [[x*timestep for x in l] for l in lengths]

    ############
    ## Output ##
    ############
    results_dir = os.path.join(out_dir, f'results/{run_name}/')
    
    print('Writing all sequences')
    if write_plots:
        plot_all_sequences(raw_pitch, time, lengths[:top_n], starts[:top_n], results_dir, clear_dir=True, plot_kwargs=plot_kwargs)
    
    if write_audio:
        write_all_sequence_audio(audio_path, starts[:top_n], lengths[:top_n], timestep, results_dir)
    
    if write_patterns:
        write_pkl(lengths[:top_n], os.path.join(results_dir, 'lengths.pkl'))
        write_pkl(starts[:top_n], os.path.join(results_dir, 'starts.pkl'))

    flush_matplotlib()

    if annotations_path:
        n_patterns = sum([len(x) for x in starts])
        coverage = get_coverage(raw_pitch, starts, lengths)
        print(f'Number of Patterns: {n_patterns}')
        print(f'Number of Groups: {len(starts_sec)}')
        print(f'Coverage: {round(coverage,2)}')
        
        annotations_raw = load_annotations_brindha(annotations_path, min_pattern_length_seconds, None)

        if s1:
            start_time = (sparse_orig_lookup[s1]*cqt_window)/sr
            end_time = (sparse_orig_lookup[s2]*cqt_window)/sr
            annotations_raw = annotations_raw[
                (annotations_raw['s1']>=start_time) & 
                (annotations_raw['s2']<=end_time)]

        recall, precision, f1, annotations = evaluate(annotations_raw, starts_sec, lengths_sec, partial_perc)
        print(f'Recall: {recall}')
        print(f'Precision: {precision}')
        print(f'F1: {f1}')
        if write_annotations:
            annotations_out_path = os.path.join(results_dir, 'annotations_tagged.csv')
            create_if_not_exists(annotations_out_path)
            annotations.to_csv(annotations_out_path, index=False)

        return recall, precision, f1
    else:
        return None, None, None

if __name__ == '__main__':
    track_name = sys.argv[1]
    run_name = sys.argv[2]
    _, _, _ = main(track_name, run_name, sr, cqt_window, s1, s2,
        gap_interp, stab_hop_secs, min_stability_length_secs, 
        freq_var_thresh_stab, conv_filter_str, bin_thresh,
        bin_thresh_segment, perc_tail,  gauss_sigma, cont_thresh,
        etc_kernel_size, binop_dim, min_diff_trav, min_pattern_length_seconds,
        min_in_group, match_tol, ext_mask_tol, n_dtw, thresh_dtw, thresh_cos,
        dupl_perc_overlap, exclusion_functions,
        annotations_path, partial_perc, top_n, write_plots, write_audio, write_patterns, write_annotations)
