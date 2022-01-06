import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq

from scipy.ndimage import label, generate_binary_structure
import skimage.io
from sklearn.cluster import DBSCAN
from itertools import groupby
from operator import itemgetter



import tqdm

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

    return y0, x0, y1, x1


def extend_segment(segment, max_l, padding):
    """
    segment: segment start and end indices [x0,x1]
    max_l: maximum indice possible
    padding: percentage to extend
    """
    x0 = segment[0]
    x1 = segment[1]
    length = x1-x0
    ext = int(padding*length)
    return [max([0,x0-ext]), min([max_l,x1+ext])]


def get_indices_of_line(l, angle, dist):
    """
    Return indices in square matrix of <l>x<l> for Hough line defined by <angle> and <dist> 
    """ 
    # Get start and end points of line to traverse from angle and dist
    x0, y0, x1, y1 = get_extremes(angle, dist, l)

    # To ensure no lines are defined outside the grid (should not be passed to func really)
    if any([y1>l, x1>l, x0>l, y0>l, y1<0, x1<0, x0<0, y0<0]):
        return None, None

    # Length of line to traverse
    length = int(np.hypot(x1-x0, y1-y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

    # x and y indices corresponding to line
    x = x.astype(int)
    y = y.astype(int)

    return x, y


def extract_segments_new(X):
    # Goal: for each thick diagonal extract one segment corresponding to pattern
    # for that iagonal
    s = generate_binary_structure(2, 2)
    labels, numL = label(X, structure=s)
    label_indices = [(labels == i).nonzero() for i in range(1, numL+1)]

    ln = 100

    all_segments = []
    for ln in tqdm.tqdm(range(len(label_indices))):
        x,y = label_indices[ln]

        points = list(zip(x,y))

        # centroid of entire diagonal
        c_x, c_y = (int(sum(x) / len(points)), int(sum(y) / len(points)))

        # split into top left quadrant and bottom right quadrant
        # since diagonal runs from top left to bottom right, this splits it in half 
        #   (top right and bottom left quadrant are empty in diagonal bounding box)
        top_half = [(x,y) for x,y in points if x <= c_x and y <= c_y]
        bottom_half = [(x,y) for x,y in points if x > c_x and y > c_y]

        top_half_x = [i[0] for i in top_half]
        top_half_y = [i[1] for i in top_half]
        bottom_half_x = [i[0] for i in bottom_half]
        bottom_half_y = [i[1] for i in bottom_half]

        # Two centroids, one for each half
        # The returned segment will pass through both of these centroids
        # splitting into two halves accounts for non 45 degree orientation
        th_cent = (int(sum(top_half_x) / len(top_half)), int(sum(top_half_y) / len(top_half)))
        bh_cent = (int(sum(bottom_half_x) / len(bottom_half)), int(sum(bottom_half_y) / len(bottom_half)))

        # Get x and y limits of bounding box of segment
        x_sorted = sorted(points, key=lambda y: x[0])
        y_sorted = sorted(points, key=lambda y: x[1])
        north_y = x_sorted[0][1]
        south_y = x_sorted[-1][1]
        west_x = y_sorted[0][0]
        east_x = y_sorted[-1][0]

        # We want the points at which the line that goes through 
        # the two centroids intersects the bounding box
        cent_points = [th_cent, bh_cent]
        x_coords, y_coords = zip(*cent_points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        # gradient and intercecpt of line passing through centroids
        m, c = lstsq(A, y_coords)[0]
        get_y = lambda x: m*x + c
        get_x = lambda y: (y - c)/m

        # does the line intersect the roof or sides of the bounding box?
        does_intersect_roof = get_y(west_x) > north_y

        if does_intersect_roof:
            y0 = north_y
            x0 = get_x(y0)

            y1 = south_y
            x1 = get_x(south_y)
        else:
            x0 = east_x
            y0 = get_y(x0)

            x1 = west_x
            y1 = get_y(west_x)

        all_segments.append([(int(x0), int(y0)), (int(x1), int(y1))])

    return all_segments


def extract_segments(matrix, angle, dist, min_diff, cqt_window, sr, padding=None):
    """
    Extract start and end coordinates of non-zero elements along hough line defined
    by <angle> and <dist>. If <padding>, extend length of each segment by <padding>%
    along the line.
    """
    # traverse hough lines and identify non-zero segments 
    l = matrix.shape[0]-1

    x, y = get_indices_of_line(l, angle, dist)
    
    # line defined outside of grid
    if x is None:
        return []

    max_l = len(x)-1

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

    if padding:
        this_segment = extend_segment(this_segment, max_l, padding)

    segments.append(this_segment)

    all_segments = []
    for i1, i2 in segments:
        # (x start, y start), (x end, y end)
        all_segments.append([(x[i1], y[i1]), (x[i2], y[i2])])

    return all_segments


def extend_segments(X_conv, X_cont, peaks, min_diff_trav, cqt_window, sr, bin_thresh_segment, perc_tail):
    """
    The segments surfaced by the convolution have tails that fade to 0, we want the binarizing
    threshold to be lower for these tails than the rest of the array
    """
    new_cont = X_cont.copy()
    all_segments = []
    for _, angle, dist in zip(*peaks):
        segments = extract_segments(X_cont, angle, dist, min_diff_trav, cqt_window, sr, padding=perc_tail)

        for s in segments:
            x0 = s[0][0]
            y0 = s[0][1]
            x1 = s[1][0]
            y1 = s[1][1]

            # Length of line to traverse
            length = int(np.hypot(x1-x0, y1-y0))
            x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

            # x and y indices corresponding to line
            x = x.astype(int)
            y = y.astype(int)
            
            for X,Y in zip(x,y):
                if X_conv[X,Y] >= bin_thresh_segment:
                    new_cont[X,Y] = 1
    return new_cont


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


def break_segment(segment_pair, mask, cqt_window, sr, timestep):
    
    # (x start, y start), (x end, y end)
    x_start = segment_pair[0][0]
    x_start_ts = int((x_start*cqt_window)/(sr*timestep))

    x_end = segment_pair[1][0]
    x_end_ts = int((x_end*cqt_window)/(sr*timestep))

    y_start = segment_pair[0][1]
    y_start_ts = int((y_start*cqt_window)/(sr*timestep))

    y_end = segment_pair[1][1]
    y_end_ts = int((y_end*cqt_window)/(sr*timestep))

    stab_x = mask[x_start_ts:x_end_ts]
    stab_y = mask[y_start_ts:y_end_ts]

    # If either sequence contains a masked region, divide
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


def break_all_segments(all_segments, mask, cqt_window, sr, timestep):
    all_broken_segments = []
    for segment_pair in all_segments:
        broken = break_segment(segment_pair, mask, cqt_window, sr, timestep)
        if len(broken) > 0:
            all_broken_segments += broken
        else:
            all_broken_segments += [segment_pair]

    return sorted([sorted(x) for x in all_broken_segments])


def do_patterns_overlap(x0, x1, y0, y1, perc_overlap):
    
    p0_indices = set(range(x0, x1+1))
    p1_indices = set(range(y0, y1+1))
    
    try:
        o1 = len(p1_indices.intersection(p0_indices))/len(p0_indices)>perc_overlap
        o2 = len(p1_indices.intersection(p0_indices))/len(p1_indices)>perc_overlap
    except:
        import ipdb; ipdb.set_trace()
    
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


# def remove_group_duplicates(group, eps):
#     start_length = sorted([(x1, x2-x1, i) for i, (x1,x2) in enumerate(group) if x2-x1>0])

#     if len(start_length) == 0:
#         return []

#     clustering = DBSCAN(eps=eps, min_samples=1)\
#                 .fit(np.array([d for d,l,i in start_length])\
#                 .reshape(-1, 1))
#     
#     with_cluster = list(zip(clustering.labels_, start_length))

#     top_indices = []
#     for g, data in groupby(with_cluster, key=itemgetter(0)):
#         data = list(data)
#         top_i = sorted(data, key=lambda y: -y[1][1])[0][1][2]
#         top_indices.append(top_i)

#     group_reduced = [x for i,x in enumerate(group) if i in top_indices]

#     return [x for x in group_reduced if x]


def remove_group_duplicates(group, perc_overlap):
    
    group_sorted = sorted(group, key= lambda y: (y[1]-y[0]), reverse=True)

    new_group = []
    skip_array = [0]*len(group_sorted)
    for i,(x0,x1) in enumerate(group_sorted):
        if skip_array[i]:
            continue
        for j,(y0,y1) in enumerate(group_sorted):
            if skip_array[j] or i==j:
                continue
            overlap = do_patterns_overlap(x0, x1, y0, y1, perc_overlap=perc_overlap)
            if overlap:
                # skip j since it is shorter
                skip_array[j] = 1
        new_group.append((x0,x1))
        skip_array[i] = 1
    return new_group


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

            ## traverse half of symmetrical array
            #if i >= j:
            #    continue

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

    rgd = [remove_group_duplicates(g, perc_overlap) for g in all_groups]
    return rgd



def group_hough_lines(peaks, hough_line_proximity=5):
    """
    Due to the nature of the convolution, actual segments are annotated by high intensity regions around
    their edges, resulting in one segment being represented by two edges, one on its uppermost and one on its 
    lowermost side. This results in two extremely close hough lines. We want to identify these
    lines and reduce them to one intersecting the region inbetween them (the locations of 
    the true segment).
    """
    # ensure indices are correct and consecutive
    peaks = (np.array(range(len(peaks[0]))), peaks[1], peaks[2])

    # Sort lines by distance from origin
    sp_ = np.array(sorted(zip(*peaks), key=lambda y: y[2]))

    sp = [sp_[::,i] for i in range(3)]

    groups = []
    this_group = []
    for i in range(len(sp_)):
        dist = sp_[i][2]
        if len(this_group)==0:
            candidate = i
            this_group.append(i)
            continue
        if abs(dist-sp_[candidate][2]) <= hough_line_proximity:
            this_group.append(i)
        else:
            groups.append(this_group)
            this_group = [i]
            candidate = i

    return [[int(sp_[i][0]) for i in g] for g in groups]


def average_hough_lines(group_indices, peaks):
    """
    Average lines ddefined in <peaks> according to indices groupings
    defined in <group_indices>
    """
    # TODO: handle variable angles, at the moment takes the first in group
    new_angles = [[peaks[1][i] for i in g][0] for g in group_indices]
    # average distances and round to nearest integer
    new_dists = [round(np.mean([peaks[2][i] for i in g][0::max(len(g)-1,1)])) for g in group_indices]

    return (np.array(range(len(new_angles))), np.array(new_angles), np.array(new_dists))


def fill_hough_groups(X, group_angles, group_dists, av_angle, av_dist):
    """
    Alter values in <X_cont> across line defined by <average> to equal 1 if
    corresponding point in any of those defined in <group> are non-zero
    """
    l = X.shape[0]-1

    X_merg = X.copy()
    
    # indices defining all lines in group
    xandy_group = [get_indices_of_line(l, angle, dist) for angle,dist in zip(group_angles, group_dists)]
    # indices defining the average line
    x_av, y_av = get_indices_of_line(l, av_angle, av_dist)

    # If any elements along lines in group are 1
    # set corresponding element in average line to 1
    for i,(x,y) in enumerate(zip(x_av, y_av)):
        for xg, yg in xandy_group:
            try:
                i_ = list(xg).index(x)
                this_x = xg[i_]
                this_y = yg[i_]
            except ValueError:
                continue
            if X[this_x, this_y]:
                for xg2, yg2 in xandy_group:
                    try:
                        i_ = list(yg2).index(y)
                        this_x2 = xg2[i_]
                        this_y2 = yg2[i_]
                    except IndexError:
                        continue
                    X_merg[this_x2, this_y2] = 1
                break

    return X_merg


def group_and_fill_hough(X, peaks, filename):
    
    X_merg = X.copy()
    
    group_indices = group_hough_lines(peaks)
    averaged_peaks = average_hough_lines(group_indices, peaks)

    peaks_angles = peaks[1]
    peaks_dists = peaks[2]

    for i in range(len(group_indices)):
        gi = group_indices[i]

        av_angle = averaged_peaks[1][i]
        av_dist = averaged_peaks[2][i]

        group_angles = peaks_angles[gi]
        group_dists = peaks_dists[gi]
        group_dists = list(range(int(min(group_dists)), int(max(group_dists))))
        group_angles = [group_angles[0]]*len(group_dists)

        X_merg = fill_hough_groups(X, group_angles, group_dists, av_angle, av_dist)

    if filename:
        skimage.io.imsave(filename, X_merg)

    return X_merg, averaged_peaks

























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




















#           sample = all_segments_reduced

#           starts_seq_exc = [[test[0][0], test[0][1]] for test in sample]
#           lengths_seq_exc = [[test[1][0]-test[0][0], test[1][1]-test[0][1]] for test in sample]

#           starts_sec_exc = [[(x*cqt_window)/(sr) for x in y] for y in starts_seq_exc]
#           lengths_sec_exc = [[(x*cqt_window)/(sr) for x in y] for y in lengths_seq_exc]

#           # Found segments broken from image processing
#           X_segments = add_segments_to_plot(X_canvas, sample)
#           skimage.io.imsave('images/segments_broken_sim_mat.png', X_segments)


#           # Patterns from full pipeline
#           X_patterns = add_patterns_to_plot(X_canvas, starts_sec_exc, lengths_sec_exc, sr, cqt_window)
#           skimage.io.imsave('images/patterns_sim_mat.png', X_patterns)








#           segment_indices = []
#           same_seqs_thresh_secs = 0.5
#           same_seqs_thresh = int(same_seqs_thresh_secs*sr/cqt_window)
#           perc_overlap = 0.8

#           all_seg_copy = all_segments_reduced.copy()

#           seg_length = lambda y: np.hypot(y[1][0]-y[0][0], y[1][1]-y[0][1])
#           all_seg_copy = sorted(all_seg_copy, key=seg_length, reverse=True)

#           all_groups = []
#           skip_array = [0]*len(all_seg_copy)
#           for i in range(len(all_seg_copy)):
#               # If this segment has been grouped already, do not consider
#               if skip_array[i] == 1:
#                   continue
#               
#               # To avoid altering original segments array
#               query_seg = all_seg_copy[i]

#               x0 = query_seg[0][0]
#               y0 = query_seg[0][1]
#               x1 = query_seg[1][0]
#               y1 = query_seg[1][1]

#               # get both corresponding patterns from query
#               this_group = [(x0,x1), (y0,y1)]
#               this_index_group = [i,i]
#               # Iterate through all other segments to identify whether
#               # any others correspond to the same two patterns...
#               for j in range(len(all_seg_copy)):

#                   # traverse half of symmetrical array
#                   #if i >= j:
#                   #    continue

#                   seg = all_seg_copy[j]

#                   # all segments correspond to two new patterns maximum...
#                   x_0 = seg[0][0]
#                   y_0 = seg[0][1]
#                   x_1 = seg[1][0]
#                   y_1 = seg[1][1]

#                   #x_match = same_seqs_marriage(x0, x1, x_0, x_1, thresh=same_seqs_thresh)
#                   #y_match = same_seqs_marriage(y0, y1, y_0, y_1, thresh=same_seqs_thresh)

#                   xx_match = do_patterns_overlap(x0, x1, x_0, x_1, perc_overlap=perc_overlap)
#                   yy_match = do_patterns_overlap(y0, y1, y_0, y_1, perc_overlap=perc_overlap)
#                   xy_match = do_patterns_overlap(x0, x1, y_0, y_1, perc_overlap=perc_overlap)
#                   yx_match = do_patterns_overlap(y0, y1, x_0, x_1, perc_overlap=perc_overlap)

#                   if (xx_match and yy_match) or (xy_match and yx_match):
#                       # same pattern, do not add and skip
#                       skip_array[j] = 1

#                   elif (xx_match and not yy_match) or (yx_match and not xy_match):
#                       # dont append the one that matches (since we already have it)
#                       if (y_0, y_1) not in this_group:
#                           this_group += [(y_0, y_1)]
#                           this_index_group += [j]
#                       skip_array[j] = 1

#                   elif (yy_match and not xx_match) or (xy_match and not yx_match):
#                       # dont append the one that matches (since we already have it)
#                       if (x_0, x_1) not in this_group:
#                           this_group += [(x_0, x_1)]
#                           this_index_group += [j]
#                       skip_array[j] = 1

#               all_groups.append(this_group)
#               segment_indices.append(this_index_group)


#           print('Convert sequences to pitch track timesteps')
#           starts_seq, lengths_seq = convert_seqs_to_timestep(all_groups, cqt_window, sr, timestep)

#           print('Applying exclusion functions')
#           #starts_seq_exc, lengths_seq_exc = apply_exclusions(raw_pitch, starts_seq, lengths_seq, exclusion_functions, min_in_group)
#           starts_seq_exc,  lengths_seq_exc = remove_below_length(starts_seq, lengths_seq, timestep, min_pattern_length_seconds)

#           starts_sec_exc = [[x*timestep for x in p] for p in starts_seq_exc]
#           lengths_sec_exc = [[x*timestep for x in l] for l in lengths_seq_exc]

#           print('Evaluating')
#           annotations_orig = load_annotations_new(annotations_path)
#           metrics = evaluate_all_tiers(annotations_orig, starts_sec_exc, lengths_sec_exc, eval_tol)


#           print('Writing all sequences')
#           plot_all_sequences(raw_pitch, time, lengths_seq_exc[:top_n], starts_seq_exc[:top_n], 'output/new_hough', clear_dir=True, plot_kwargs=plot_kwargs)
#           write_all_sequence_audio(audio_path, starts_seq_exc[:top_n], lengths_seq_exc[:top_n], timestep, 'output/new_hough')





#           # plot one group
#           i=0
#           j=2
#           if j:
#               sse = [starts_sec_exc[i], starts_sec_exc[j]]
#               lse = [lengths_sec_exc[i], lengths_sec_exc[j]]
#               indices1 = segment_indices[i]
#               indices2 = segment_indices[j]
#           else:
#               sse = [starts_sec_exc[i]]
#               lse = [lengths_sec_exc[i]]
#               indices1 = segment_indices[i]
#               indices2 = None

#           X_patterns = add_patterns_to_plot(X_canvas, sse, lse, sr, cqt_window)
#           skimage.io.imsave('images/patterns_sim_mat.png', X_patterns)
#           print(indices1)
#           print(indices2)





#           segmentj = all_segments_reduced[0]
#           segmenti = all_segments_reduced[4]

#           x0 = segmenti[0][0]
#           y0 = segmenti[0][1]
#           x1 = segmenti[1][0]
#           y1 = segmenti[1][1]

#           x_0 = segmentj[0][0]
#           y_0 = segmentj[0][1]
#           x_1 = segmentj[1][0]
#           y_1 = segmentj[1][1]

#           xx_match = do_patterns_overlap(x0, x1, x_0, x_1, perc_overlap=0.7)
#           yy_match = do_patterns_overlap(y0, y1, y_0, y_1, perc_overlap=0.7)
#           xy_match = do_patterns_overlap(x0, x1, y_0, y_1, perc_overlap=0.7)
#           yx_match = do_patterns_overlap(y0, y1, x_0, x_1, perc_overlap=0.7)

#           # Found segments broken from image processing
#           X_segments = add_segments_to_plot(X_canvas, [segmenti, segmentj])
#           skimage.io.imsave('images/segments_broken_sim_mat.png', X_segments)







#           # do_patterns_overlap
#           p0_indices = set(range(x0, x1+1))
#           p1_indices = set(range(y_0, y_1+1))

#           o1 = len(p1_indices.intersection(p0_indices))/len(p0_indices)>perc_overlap
#           o2 = len(p1_indices.intersection(p0_indices))/len(p1_indices)>perc_overlap








