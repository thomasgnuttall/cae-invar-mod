import numpy as np

from sklearn.cluster import DBSCAN
from itertools import groupby
from operator import itemgetter

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


def extract_segments(matrix, angle, dist, min_diff, cqt_window, sr):
    # traverse hough lines and identify non-zero segments 
    l = matrix.shape[0]-1

    # Get start and end points of line to traverse from angle and dist
    x0, y0, x1, y1 = get_extremes(angle, dist, l)

    # To ensure no lines are defined outside the grid (should not be passed to func really)
    if any([y1>l, x1>l, x0>l, y0>l, y1<0, x1<0, x0<0, y0<0]):
        return []

    # Length of line to traverse
    length = int(np.hypot(x1-x0, y1-y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

    # x and y indices corresponding to line
    x = x.astype(int)
    y = y.astype(int)

    # Extract the values along the line
    zi = matrix[x, y]

    # Get index of non-zero elements along line
    non_zero = np.where(zi != 0)[0]
    if len(non_zero) == 0:
        return []

    # Identify segments of continuous non-zero along line
    segments = []
    this_segment = []
    for i in range(len(non_zero)):
        # First non zero must be a start point of segment
        if i == 0:
            this_segment.append(non_zero[i])
            continue

        # Number of elements along hypotenuse
        n_elems = non_zero[i] - non_zero[i-1]

        # Time corresponding to gap found between this silence and previous
        #   - n_elems is length of hypotonuse in cqt space
        #   - (assume equilateral) divide by sqrt(2) to get adjacent length (length of gap)
        #   - mulitply by cqt_window and divide by sample rate to get adjacent length in seconds
        T = (cqt_window * n_elems) / (sr * 2**0.5)

        # If gap is smaller than min_diff, ignore it
        if T <= min_diff:
            continue
        else:
            # consider gap the end of found segment and store
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


def get_all_segments(X, peaks, min_diff_trav, min_length_cqt, cqt_window, sr):
    all_segments = []
    for _, angle, dist in zip(*peaks):
        segments = extract_segments(X, angle, dist, min_diff_trav, cqt_window, sr)

        # If either of the lengths are above minimum length, add to all segments
        for s in segments:
            x0 = s[0][0]
            y0 = s[0][1]
            x1 = s[1][0]
            y1 = s[1][1]

            l0 = x1-x0
            l1 = y1-y0

            # temp | to_add = []
            # temp | if max([l1, l0]) > min_length_cqt:
                # temp | to_add.append(s)

            all_segments.append(s)

        # temp | all_segments += to_add

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


def do_patterns_overlap(x0, x1, y0, y1, perc_overlap):
    
    p0_indices = set(range(x0, x1+1))
    p1_indices = set(range(y0, y1+1))
    
    o1 = len(p1_indices.intersection(p0_indices))/len(p0_indices)>perc_overlap
    o2 = len(p1_indices.intersection(p0_indices))/len(p1_indices)>perc_overlap
    
    return o1 and o2


def do_segments_overlap(seg1, seg2, perc_overlap=0.5):
    """
    The Hough transform allows for the same segment to be intersected
    twice by lines of slightly different angle. We want to take the 
    longest of these duplicates and discard the rest
    
    Two segments inputed
      ...each formatted - [(x0, y0), (x1, y1)]
    These patterns could be distinct regions or not
    """
    # Assign the longest of the two segments to 
    # segment L and the shortest to segment S
    (x0, y0), (x1, y1) = seg1
    (x2, y2), (x3, y3) = seg2

    len_seg1 = np.hypot(x1-x0, y1-y0)
    len_seg2 = np.hypot(x3-x2, y3-y2)

    if len_seg1 >= len_seg2:
        seg_L = seg1
        seg_S = seg2
    else:
        seg_L = seg2
        seg_S = seg1

    # Each segment corresponds to two patterns
    #   - [segment 1] p0: x0 -> x1
    #   - [segment 1] p1: y0 -> y1
    #   - [segment 2] p2: x2 -> x3
    #   - [segment 2] p3: y2 -> y3

    (lx0, ly0), (lx1, ly1) = seg_L
    (sx0, sy0), (sx1, sy1) = seg_S

    # The two segments correspond to the same pair of patterns 
    # if p2 is a subset of p0 AND p3 is a subset of p2
    # We consider "subset" to mean > <perc_overlap>% overlap in indices
    overlap1 = do_patterns_overlap(lx0, lx1, sx0, sx1, perc_overlap=perc_overlap)
    overlap2 = do_patterns_overlap(ly0, ly1, sy0, sy1, perc_overlap=perc_overlap)

    # Return True if overlap in both dimensions
    return overlap1 and overlap2


def reduce_duplicates(all_segments, perc_overlap=0.5):
    all_seg_copy = all_segments.copy()

    # Order by length to speed computation
    seg_length = lambda y: np.hypot(y[1][0]-y[0][0], y[1][1]-y[0][1])
    all_seg_copy = sorted(all_seg_copy, key=seg_length, reverse=True)

    skip_array = [0]*len(all_seg_copy)
    reduced_segments = []
    # Iterate through all patterns and remove duplicates
    for i, seg1 in enumerate(all_seg_copy):
        # If this segment has been grouped already, do not consider
        if skip_array[i] == 1:
            continue

        for j, seg2 in enumerate(all_seg_copy[i+1:], i+1):
            # True or False, do they overlap in x and y?
            overlap = do_segments_overlap(seg1, seg2, perc_overlap=perc_overlap)

            # If they overlap discard seg2 (since it is shorter)
            if overlap:
                # remove this pattern
                skip_array[j] = 1

        # append original pattern
        reduced_segments += [seg1]

    return reduced_segments


def remove_short(all_segments, min_length_cqt):
    long_segs = []
    for (x0, y0), (x1, y1) in all_segments:
        length1 = x1 - x0
        length2 = y1 - y0
        if any([length1>min_length_cqt, length2>min_length_cqt]):
            long_segs.append([(x0, y0), (x1, y1)])
    return long_segs


def same_seqs_marriage(x1, y1, x2, y2, thresh=4):
    return (abs(x1-x2) < thresh) and (abs(y1-y2) < thresh)


def get_length(x1,y1,x2,y2):
    return ((x2-x1)**2 + (y2-y1)**2)**0.5


def remove_group_duplicates(group, eps):
    start_length = sorted([(x1, x2-x1, i) for i, (x1,x2) in enumerate(group) if x2-x1>0])

    if len(start_length) == 0:
        return []

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

    return [x for x in group_reduced if x]


def get_longest(x0,x1,y0,y1):
    len_x = x1-x0
    len_y = y1-y0
    if len_x > len_y:
        return x0, x1
    else:
        return y0, y1


def group_segments(all_segments, perc_overlap):
    all_seg_copy = all_segments.copy()

    seg_length = lambda y: np.hypot(y[1][0]-y[0][0], y[1][1]-y[0][1])
    all_seg_copy = sorted(all_seg_copy, key=seg_length, reverse=True)

    all_groups = []
    skip_array = [0]*len(all_seg_copy)
    for i in range(len(all_seg_copy)):
        # If this segment has been grouped already, do not consider
        if skip_array[i] == 1:
            continue
        
        # To avoid altering original segments array
        query_seg = all_seg_copy[i]

        x0 = query_seg[0][0]
        y0 = query_seg[0][1]
        x1 = query_seg[1][0]
        y1 = query_seg[1][1]

        # get both corresponding patterns from query
        this_group = [(x0,x1), (y0,y1)]

        # Iterate through all other segments to identify whether
        # any others correspond to the same two patterns...
        for j in range(len(all_seg_copy)):

            # traverse half of symmetrical array
            if i >= j:
                continue

            seg = all_seg_copy[j]

            x_0 = seg[0][0]
            y_0 = seg[0][1]
            x_1 = seg[1][0]
            y_1 = seg[1][1]

            #x_match = same_seqs_marriage(x0, x1, x_0, x_1, thresh=same_seqs_thresh)
            #y_match = same_seqs_marriage(y0, y1, y_0, y_1, thresh=same_seqs_thresh)

            xx_match = do_patterns_overlap(x0, x1, x_0, x_1, perc_overlap=perc_overlap)
            yy_match = do_patterns_overlap(y0, y1, y_0, y_1, perc_overlap=perc_overlap)
            xy_match = do_patterns_overlap(x0, x1, y_0, y_1, perc_overlap=perc_overlap)
            yx_match = do_patterns_overlap(y0, y1, x_0, x_1, perc_overlap=perc_overlap)

            if (xx_match and yy_match) or (xy_match and yx_match):
                # same pattern, do not add and skip
                skip_array[j] = 1

            elif (xx_match and not yy_match) or (yx_match and not xy_match):
                # dont append the one that matches (since we already have it)
                if (y_0, y_1) not in this_group:
                    this_group += [(y_0, y_1)]
                skip_array[j] = 1

            elif (yy_match and not xx_match) or (xy_match and not yx_match):
                # dont append the one that matches (since we already have it)
                if (x_0, x_1) not in this_group:
                    this_group += [(x_0, x_1)]
                skip_array[j] = 1

        all_groups.append(this_group)

    rgd = [remove_group_duplicates(g, eps=5) for g in all_groups]
    return rgd




# query_seg
# [(1799, 2040), (1951, 2192)]

# all_seg_copy[4]
# [(2025, 2263), (2160, 2398)]

# all_seg_copy[5]
# [(2263, 2025), (2398, 2160)]

# xx_match = False
# yy_match = False
# xy_match = True
# yx_match = True


# x0 = 1799
# x1 = 1951
# y0 = 2040
# y1 = 2192

# x_0 = 2279
# x_1 = 2313
# y_0 = 1195
# y_1 = 1229


# xx_match = do_patterns_overlap(x0, x1, x_0, x_1, perc_overlap=dupl_perc_overlap)
# yy_match = do_patterns_overlap(y0, y1, y_0, y_1, perc_overlap=dupl_perc_overlap)
# xy_match = do_patterns_overlap(x0, x1, y_0, y_1, perc_overlap=dupl_perc_overlap)
# yx_match = do_patterns_overlap(y0, y1, x_0, x_1, perc_overlap=dupl_perc_overlap)




















sample = all_segments_reduced

starts_seq_exc = [[test[0][0], test[0][1]] for test in sample]
lengths_seq_exc = [[test[1][0]-test[0][0], test[1][1]-test[0][1]] for test in sample]

starts_sec_exc = [[(x*cqt_window)/(sr) for x in y] for y in starts_seq_exc]
lengths_sec_exc = [[(x*cqt_window)/(sr) for x in y] for y in lengths_seq_exc]

# Found segments broken from image processing
X_segments = add_segments_to_plot(X_canvas, sample)
skimage.io.imsave('images/segments_broken_sim_mat.png', X_segments)


# Patterns from full pipeline
X_patterns = add_patterns_to_plot(X_canvas, starts_sec_exc, lengths_sec_exc, sr, cqt_window)
skimage.io.imsave('images/patterns_sim_mat.png', X_patterns)








segment_indices = []
same_seqs_thresh_secs = 0.5
same_seqs_thresh = int(same_seqs_thresh_secs*sr/cqt_window)
perc_overlap = 0.8

all_seg_copy = all_segments_reduced.copy()

seg_length = lambda y: np.hypot(y[1][0]-y[0][0], y[1][1]-y[0][1])
all_seg_copy = sorted(all_seg_copy, key=seg_length, reverse=True)

all_groups = []
skip_array = [0]*len(all_seg_copy)
for i in range(len(all_seg_copy)):
    # If this segment has been grouped already, do not consider
    if skip_array[i] == 1:
        continue
    
    # To avoid altering original segments array
    query_seg = all_seg_copy[i]

    x0 = query_seg[0][0]
    y0 = query_seg[0][1]
    x1 = query_seg[1][0]
    y1 = query_seg[1][1]

    # get both corresponding patterns from query
    this_group = [(x0,x1), (y0,y1)]
    this_index_group = [i,i]
    # Iterate through all other segments to identify whether
    # any others correspond to the same two patterns...
    for j in range(len(all_seg_copy)):

        # traverse half of symmetrical array
        if i >= j:
            continue

        seg = all_seg_copy[j]

        # all segments correspond to two new patterns maximum...
        x_0 = seg[0][0]
        y_0 = seg[0][1]
        x_1 = seg[1][0]
        y_1 = seg[1][1]

        #x_match = same_seqs_marriage(x0, x1, x_0, x_1, thresh=same_seqs_thresh)
        #y_match = same_seqs_marriage(y0, y1, y_0, y_1, thresh=same_seqs_thresh)

        xx_match = do_patterns_overlap(x0, x1, x_0, x_1, perc_overlap=perc_overlap)
        yy_match = do_patterns_overlap(y0, y1, y_0, y_1, perc_overlap=perc_overlap)
        xy_match = do_patterns_overlap(x0, x1, y_0, y_1, perc_overlap=perc_overlap)
        yx_match = do_patterns_overlap(y0, y1, x_0, x_1, perc_overlap=perc_overlap)

        if (xx_match and yy_match) or (xy_match and yx_match):
            # same pattern, do not add and skip
            skip_array[j] = 1

        elif (xx_match and not yy_match) or (yx_match and not xy_match):
            # dont append the one that matches (since we already have it)
            if (y_0, y_1) not in this_group:
                this_group += [(y_0, y_1)]
                this_index_group += [j]
            skip_array[j] = 1

        elif (yy_match and not xx_match) or (xy_match and not yx_match):
            # dont append the one that matches (since we already have it)
            if (x_0, x_1) not in this_group:
                this_group += [(x_0, x_1)]
                this_index_group += [j]
            skip_array[j] = 1

    all_groups.append(this_group)
    segment_indices.append(this_index_group)


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


print('Writing all sequences')
plot_all_sequences(raw_pitch, time, lengths_seq_exc[:top_n], starts_seq_exc[:top_n], 'output/new_hough', clear_dir=True, plot_kwargs=plot_kwargs)
write_all_sequence_audio(audio_path, starts_seq_exc[:top_n], lengths_seq_exc[:top_n], timestep, 'output/new_hough')





# plot one group
i=0
j=2
if j:
    sse = [starts_sec_exc[i], starts_sec_exc[j]]
    lse = [lengths_sec_exc[i], lengths_sec_exc[j]]
    indices1 = segment_indices[i]
    indices2 = segment_indices[j]
else:
    sse = [starts_sec_exc[i]]
    lse = [lengths_sec_exc[i]]
    indices1 = segment_indices[i]
    indices2 = None

X_patterns = add_patterns_to_plot(X_canvas, sse, lse, sr, cqt_window)
skimage.io.imsave('images/patterns_sim_mat.png', X_patterns)
print(indices1)
print(indices2)





segmentj = all_segments_reduced[0]
segmenti = all_segments_reduced[4]

x0 = segmenti[0][0]
y0 = segmenti[0][1]
x1 = segmenti[1][0]
y1 = segmenti[1][1]

x_0 = segmentj[0][0]
y_0 = segmentj[0][1]
x_1 = segmentj[1][0]
y_1 = segmentj[1][1]

xx_match = do_patterns_overlap(x0, x1, x_0, x_1, perc_overlap=0.7)
yy_match = do_patterns_overlap(y0, y1, y_0, y_1, perc_overlap=0.7)
xy_match = do_patterns_overlap(x0, x1, y_0, y_1, perc_overlap=0.7)
yx_match = do_patterns_overlap(y0, y1, x_0, x_1, perc_overlap=0.7)

# Found segments broken from image processing
X_segments = add_segments_to_plot(X_canvas, [segmenti, segmentj])
skimage.io.imsave('images/segments_broken_sim_mat.png', X_segments)







# do_patterns_overlap
p0_indices = set(range(x0, x1+1))
p1_indices = set(range(y_0, y_1+1))

o1 = len(p1_indices.intersection(p0_indices))/len(p0_indices)>perc_overlap
o2 = len(p1_indices.intersection(p0_indices))/len(p1_indices)>perc_overlap








