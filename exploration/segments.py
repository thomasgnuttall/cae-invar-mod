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


def get_label_indices(X, structure_size=2):
    s = generate_binary_structure(structure_size, structure_size)
    labels, numL = label(X, structure=s)
    label_indices = [(labels == i).nonzero() for i in range(1, numL+1)]
    return label_indices


def line_through_points(x0, y0, x1, y1):
    """
    return function to convert x->y and for y->x
    for straight line that passes through x0,y0 and x1,y1
    """
    centroids = [(x0,y0), (x1, y1)]
    x_coords, y_coords = zip(*centroids)
    
    # gradient and intercecpt of line passing through centroids
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]

    # functions for converting between
    # x and y on that line
    get_y = lambda xin: m*xin + c
    get_x = lambda yin: (yin - c)/m

    return get_x, get_y


def extract_segments_new(X):
    
    # size of square array
    n = X.shape[0]

    # Goal: for each thick diagonal extract one segment 
    # corresponding to pattern for that diagonal
    label_indices = get_label_indices(X)

    all_segments = []
    for ix, (x,y) in enumerate(label_indices):
        # (x,y) indices of points that define this diagonal.
        # These correspond to an area more than 1 element thick,
        # the objective is to identify a single path through
        # this area to nominate as the candidate underlying segment/pattern
        points = list(zip(x,y))

        # centroid of entire diagonal
        c_x, c_y = (int(sum(x) / len(points)), int(sum(y) / len(points)))

        # split into top left quadrant (tlq) and bottom right quadrant (brq) 
        #   (top right and bottom left quadrant are empty in diagonal bounding box)
        tlq = [(x,y) for x,y in points if x <= c_x and y <= c_y]
        brq = [(x,y) for x,y in points if x > c_x and y > c_y]

        tlq_x = [i[0] for i in tlq]
        tlq_y = [i[1] for i in tlq]

        brq_x = [i[0] for i in brq]
        brq_y = [i[1] for i in brq]

        # Compute the centroid for each of the two quarters
        tlq_centroid = (int(sum(tlq_x) / len(tlq)), int(sum(tlq_y) / len(tlq)))
        brq_centroid = (int(sum(brq_x) / len(brq)), int(sum(brq_y) / len(brq)))

        # Get x and y limits of bounding box of entire area
        x_sorted = sorted(points, key=lambda y: y[0])
        y_sorted = sorted(points, key=lambda y: y[1])
        
        north_y = y_sorted[0][1] # line across the top
        south_y = y_sorted[-1][1] # line across the bottom
        west_x  = x_sorted[0][0] # line across left side
        east_x  = x_sorted[-1][0] # line across right side

        # functions for converting between
        # x and y on that line
        get_x, get_y = line_through_points(
            tlq_centroid[0], tlq_centroid[1], brq_centroid[0], brq_centroid[1])

        # does the line intersect the roof or sides of the bounding box?
        does_intersect_roof = get_y(west_x) > north_y

        if does_intersect_roof:
            y0 = north_y
            x0 = get_x(y0)

            y1 = south_y
            x1 = get_x(y1)
        else:
            x0 = west_x
            y0 = get_y(x0)

            x1 = east_x
            y1 = get_y(x1)
        
        # int() always rounds down
        roundit = lambda yin: int(round(yin))

        # Points are computed using a line learnt
        # using least squares, there is a small chance that 
        # this results in one of the coordinates being slightly
        # over the limits of the array, the rounding that occurs 
        # when converting to int may make this +/- 1 outside of array
        # limits
        if roundit(x0) < 0:
            x0 = 0
            y0 = roundit(get_y(0))
        if roundit(y0) < 0:
            y0 = 0
            x0 = roundit(get_x(0))
        if roundit(x1) >= n:
            x1 = n-1
            y1 = roundit(get_y(x1))
        if roundit(y1) >= n:
            y1 = n-1
            x1 = roundit(get_x(y1))
        
        # some 
        if not any([roundit(x1) < roundit(x0), roundit(y1) < roundit(y1)]):
            all_segments.append([(roundit(x0), roundit(y0)), (roundit(x1), roundit(y1))])

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


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def extend_segments(all_segments, X_in, X_conv, perc_tail, bin_thresh_segment):
    h,w = X_in.shape

    all_segments_extended = []
    for ((x0, y0), (x1, y1)) in all_segments:    
        dx = (x1-x0)
        dy = (y1-y0)

        length = (dx**2 + dy**2)**0.5
        grad = dy/dx

        # Angle line makes with x axis
        theta = np.arctan(grad)

        # Length of extra part of line
        extra = length * perc_tail

        # Get new start and end points of extended segment
        Dx = extra/np.sin(theta)
        Dy = extra/np.cos(theta) 

        X0 = int(x0 - Dx)
        Y0 = int(y0 - Dy)
        X1 = int(x1 + Dx)
        Y1 = int(y1 + Dy)

        # Coordinates of line connecting X0, Y0 and X1, Y1
        new_length = round(length + 2*extra)
        X, Y = np.linspace(X0, X1, new_length), np.linspace(Y0, Y1, new_length)
        X = [round(x) for x in X]
        Y = [round(y) for y in Y]
        filts = [(x,y) for x,y in zip(X,Y) if all([x>=0, y>=0, x<w, y<h])]
        X = [x[0] for x in filts]
        Y = [x[1] for x in filts]

        # the line can be cut short because of matrix boundaries
        clos0 = closest_node((x0,y0), list(zip(X,Y)))
        clos1 = closest_node((x1,y1), list(zip(X,Y)))

        new_seg = X_conv[X,Y]>bin_thresh_segment
        # original segment is always 1
        new_seg[clos0:clos1+1] = 1

        i0 = clos0
        i1 = clos1
        # go backwards through preceeding extension until there are no more
        # values that correspond to similarity above threshold
        for i,v in list(enumerate(new_seg))[:clos0][::-1]:
            if v == 0:
                i0 = i + 1
                break

        # go forwards through succeeding extension until there are no more
        # values that correspond to similarity above threshold
        for i,v in list(enumerate(new_seg))[clos1:]:
            if v == 0:
                i1 = i - 1
                break

        x0_new = X[i0]
        y0_new = Y[i0]

        x1_new = X[i1]
        y1_new = Y[i1]

        ext_segment = [(x0_new, y0_new), (x1_new, y1_new)]
        all_segments_extended.append(ext_segment)

    return all_segments_extended


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
    x_start_ts = round((x_start*cqt_window)/(sr*timestep))

    x_end = segment_pair[1][0]
    x_end_ts = round((x_end*cqt_window)/(sr*timestep))

    y_start = segment_pair[0][1]
    y_start_ts = round((y_start*cqt_window)/(sr*timestep))

    y_end = segment_pair[1][1]
    y_end_ts = round((y_end*cqt_window)/(sr*timestep))

    stab_x = mask[x_start_ts:x_end_ts]
    stab_y = mask[y_start_ts:y_end_ts]

    # If either sequence contains a masked region, divide
    if any([2 in stab_x, 2 in stab_y]):
        break_points_x = np.where(stab_x==2)[0]
        break_points_y = np.where(stab_y==2)[0]
        if len(break_points_y) > len(break_points_x):
            bpy_ = break_points_y
            # break points x should correspond to the same proportion through the sequence
            # as break points y, since they can be different lengths
            bpx_ = [round((b/len(stab_y))*len(stab_x)) for b in bpy_]
            
            # convert back to cqt_window granularity sequence
            bpx = [round(x*(sr*timestep)/cqt_window) for x in bpx_]
            bpy = [round(y*(sr*timestep)/cqt_window) for y in bpy_]
        else:
            bpx_ = break_points_x
            # break points y should correspond to the same proportion through the sequence
            # as break points x, since they can be different lengths
            bpy_ = [round((b/len(stab_x))*len(stab_y)) for b in bpx_]
            
            # convert back to cqt_window granularity sequence
            bpy = [round(x*(sr*timestep)/cqt_window) for x in bpy_]
            bpx = [round(x*(sr*timestep)/cqt_window) for x in bpx_]
    else:
        # nothing to be broken, return original segment
        return [[(x_start, y_start), (x_end, y_end)]]

    new_segments = []
    for i in range(len(bpx)):
        bx = bpx[i]
        by = bpy[i]

        if i == 0:
            new_segments.append([(x_start, y_start), (x_start+bx, y_start+by)])
        else:
            # break points from last iterations
            # we begin on these this time
            bx1 = bpx[i-1]
            by1 = bpy[i-1]

            new_segments.append([(x_start+bx1, y_start+by1), (x_start+bx, y_start+by)])

    new_segments.append([(x_start+bx, y_start+by), (x_end, y_end)])

    return new_segments


def break_all_segments(all_segments, mask, cqt_window, sr, timestep):
    all_broken_segments = []
    for segment_pair in all_segments:
        broken = break_segment(segment_pair, mask, cqt_window, sr, timestep)
        # if there is nothing to break, the 
        # original segment pair is returned
        all_broken_segments += broken
    return sorted([sorted(x) for x in all_broken_segments])


def get_overlap(x0, x1, y0, y1):
    
    p0_indices = set(range(x0, x1+1))
    p1_indices = set(range(y0, y1+1))
    
    inters = p1_indices.intersection(p0_indices)

    o1 = len(inters)/len(p0_indices)
    o2 = len(inters)/len(p1_indices)
    
    return o1, o2


def do_patterns_overlap(x0, x1, y0, y1, perc_overlap):
    
    o1, o2 = get_overlap(x0, x1, y0, y1)

    return o1>perc_overlap and o2>perc_overlap


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
        if all([length1>min_length_cqt, length2>min_length_cqt]):
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


def is_good_segment(x0, y0, x1, y1, thresh, silence_and_stable_mask, cqt_window, timestep, sr):
    x0s = round(x0*cqt_window/(sr*timestep))
    x1s = round(x1*cqt_window/(sr*timestep))
    y0s = round(y0*cqt_window/(sr*timestep))
    y1s = round(y1*cqt_window/(sr*timestep))

    seq1_stab = silence_and_stable_mask[x0s:x1s]
    seq2_stab = silence_and_stable_mask[y0s:y1s]
    
    prop_stab1 = sum(seq1_stab!=0) / len(seq1_stab)
    prop_stab2 = sum(seq2_stab!=0) / len(seq2_stab)

    if not (prop_stab1 > 0.6 or prop_stab2 > 0.6):
        return True
    else:
        return False


def matches_dict_to_groups(matches_dict):
    all_groups = []
    c=0
    for i, matches in matches_dict.items():
        this_group = [i] + matches
        for j,ag in enumerate(all_groups):
            if set(this_group).intersection(set(ag)):
                # group exists, append
                all_groups[j] = list(set(all_groups[j] + this_group))
                c = 1
                break
        if c==0:
            # group doesnt exist yet
            all_groups.append(this_group)
        c=0
    return all_groups


def check_groups_unique(all_groups):
    repl = True
    for i,ag in enumerate(all_groups):
        for j,ag1 in enumerate(all_groups):
            if i==j:
                continue
            if set(ag).intersection(set(ag1)):
                print(f"groups {i} and {j} intersect")
                repl = False
    return repl


def compare_segments(i, j, Qx0, Qy0, Qx1, Qy1, Rx0, Ry0, Rx1, Ry1, min_length_cqt, all_new_segs, max_i, matches_dict):
    """
    # Types of matches for two sequences: 
    #       query (Q):(------) and returned (R):[-------]
    # 1. (------) [------] - no match
    #   - Do nothing
    # 2. (-----[-)-----] - insignificant overlap
    #   - Do nothing
    # 3. (-[------)-] - left not significant, overlap significant, right not significant
    #   - Group Q and R


    # Query is on the left: Qx0 < Rx0
    #################################
    # 4. (-[-------)--------] - left not significant, overlap significant, right significant
    #   - Cut R to create R1 and R2 (where R1+R2 = R)
    #   - Add R1 and Q to group
    #   - R2 and R1 marked as new segments
    # 5. (---------[------)-] - left significant, overlap significant, right not significant
    #   - Cut Q to create Q1 and Q2 (where Q1+Q2 = Q)
    #   - Add Q2 and R to group
    #   - Q1 and Q2 marked as new segments
    # 6. (---------[------)-------] - left significant, overlap significant, right significant
    #   - cut Q to create Q1 and Q2 (where Q1+Q2 = Q)
    #   - cut R to create R1 and R2 (where R1+R2 = R)
    #   - Add Q2 and R1 to group
    #   - Q1, Q2, R1 and R2 marked as new segments


    # Query is on the left: Rx0 < Qx0
    #################################
    # 7. [-(-------]--------) - left not significant, overlap significant, right significant
    #   - Cut Q to create Q1 and Q2 (where Q1+Q2 = Q)
    #   - Add R and Q1 to group
    #   - Q1 and Q2 marked as new segments
    # 8. [---------(------]-) - left significant, overlap significant, right not significant
    #   - Cut R to create R1 and R2 (where R1+R2 = R)
    #   - Add R2 and Q to group
    #   - R1 and R2 marked as new segments
    # 9. [---------(------]-------) - left significant, overlap significant, right significant
    #   - cut Q to create Q1 and Q2 (where Q1+Q2 = Q)
    #   - cut R to create R1 and R2 (where R1+R2 = R)
    #   - Add R2 and Q1 to group
    #   - Q1, Q2, R1 and R2 marked as new segments

    """

    # functions that define line through query(Q) segment
    Qget_x, Qget_y = line_through_points(Qx0, Qy0, Qx1, Qy1)
    # get indices corresponding to query(Q)
    Q_indices = set(range(Qx0, Qx1+1))

    # functions that define line through returned(R) segment
    Rget_x, Rget_y = line_through_points(Rx0, Ry0, Rx1, Ry1)
    # get indices corresponding to query(Q)
    R_indices = set(range(Rx0, Rx1+1))


    # query on the left
    if Qx0 <= Rx0:
        # indices in common between query(Q) and returned(R)
        left_indices = Q_indices.difference(R_indices)
        overlap_indices = Q_indices.intersection(R_indices)
        right_indices = R_indices.difference(Q_indices)

        # which parts in the venn diagram
        # betweem Q and R are large enough to
        # be considered
        left_sig = len(left_indices) >= min_length_cqt
        overlap_sig = len(overlap_indices) >= min_length_cqt
        right_sig = len(right_indices) >= min_length_cqt

        # which type of match (if any). 
        # See above for explanation
        type_1 = not overlap_indices
        type_2 = not overlap_sig and overlap_indices
        type_3 = all([overlap_sig, not left_sig, not right_sig])

        type_4 = all([not left_sig, overlap_sig, right_sig])
        type_5 = all([left_sig, overlap_sig, not right_sig])
        type_6 = all([left_sig, overlap_sig, right_sig])

        type_7 = False
        type_8 = False
        type_9 = False

    # query on the right
    if Rx0 < Qx0:
        # indices in common between query(Q) and returned(R)
        left_indices = R_indices.difference(Q_indices)
        overlap_indices = R_indices.intersection(Q_indices)
        right_indices = Q_indices.difference(R_indices)

        # which parts in the venn diagram
        # betweem Q and R are large enough to
        # be considered
        left_sig = len(left_indices) >= min_length_cqt
        overlap_sig = len(overlap_indices) >= min_length_cqt
        right_sig = len(right_indices) >= min_length_cqt

        # which type of match (if any). 
        # See above for explanation
        type_1 = not overlap_indices
        type_2 = not overlap_sig and overlap_indices
        type_3 = all([overlap_sig, not left_sig, not right_sig])

        type_4 = False
        type_5 = False
        type_6 = False

        type_7 = all([not left_sig, overlap_sig, right_sig])
        type_8 = all([left_sig, overlap_sig, not right_sig])
        type_9 = all([left_sig, overlap_sig, right_sig])

    if type_3:
        # record match, no further action
        update_dict(matches_dict, i, j)
        update_dict(matches_dict, j, i)

    if type_4:

        # Split R into two patterns
        # that which intersects with
        # Q....
        R1x0 = min(overlap_indices)
        R1x1 = max(overlap_indices)
        R1y0 = round(Rget_y(R1x0)) # extrapolate segment for corresponding y
        R1y1 = round(Rget_y(R1x1)) # extrapolate segment for corresponding y
        R1_seg = ((R1x0, R1y0), (R1x1, R1y1))

        # And that part which does 
        # not intersect with Q...
        R2x0 = min(right_indices)
        R2x1 = max(right_indices)
        R2y0 = round(Rget_y(R2x0)) # extrapolate segment for corresponding y
        R2y1 = round(Rget_y(R2x1)) # extrapolate segment for corresponding y
        R2_seg = ((R2x0, R2y0), (R2x1, R2y1))
        
        # Log new R1 seg and group with Q
        max_i += 1
        all_new_segs.append(R1_seg)
        update_dict(matches_dict, i, max_i)
        update_dict(matches_dict, max_i, i)
        
        # Log new R2 seg
        max_i += 1
        all_new_segs.append(R2_seg)

    if type_5:

        # Split Q into two patterns
        # that which does not intersects
        # with R....
        Q1x0 = min(left_indices)
        Q1x1 = max(left_indices)
        Q1y0 = round(Qget_y(Q1x0)) # extrapolate segment for corresponding y
        Q1y1 = round(Qget_y(Q1x1)) # extrapolate segment for corresponding y
        Q1_seg = ((Q1x0, Q1y0), (Q1x1, Q1y1))

        # And that part which does 
        # intersect with R...
        Q2x0 = min(overlap_indices)
        Q2x1 = max(overlap_indices)
        Q2y0 = round(Qget_y(Q2x0)) # extrapolate segment for corresponding y
        Q2y1 = round(Qget_y(Q2x1)) # extrapolate segment for corresponding y
        Q2_seg = ((Q2x0, Q2y0), (Q2x1, Q2y1))
        
        # Log new Q2 seg and group with R
        max_i += 1
        all_new_segs.append(Q2_seg)
        update_dict(matches_dict, j, max_i)
        update_dict(matches_dict, max_i, j)
        
        # Log new Q1 seg
        max_i += 1
        all_new_segs.append(Q1_seg)

    if type_6:
        
        # Split Q into two patterns
        # that which does not intersect
        #  with R....
        Q1x0 = min(left_indices)
        Q1x1 = max(left_indices)
        Q1y0 = round(Qget_y(Q1x0)) # extrapolate segment for corresponding y
        Q1y1 = round(Qget_y(Q1x1)) # extrapolate segment for corresponding y
        Q1_seg = ((Q1x0, Q1y0), (Q1x1, Q1y1))

        # And that part which does 
        # intersect with R...
        Q2x0 = min(overlap_indices)
        Q2x1 = max(overlap_indices)
        Q2y0 = round(Qget_y(Q2x0)) # extrapolate segment for corresponding y
        Q2y1 = round(Qget_y(Q2x1)) # extrapolate segment for corresponding y
        Q2_seg = ((Q2x0, Q2y0), (Q2x1, Q2y1)) 

        # Split R into two patterns
        # that which intersects with
        # Q....
        R1x0 = min(overlap_indices)
        R1x1 = max(overlap_indices)
        R1y0 = round(Rget_y(R1x0)) # extrapolate segment for corresponding y
        R1y1 = round(Rget_y(R1x1)) # extrapolate segment for corresponding y
        R1_seg = ((R1x0, R1y0), (R1x1, R1y1))

        # And that part which does 
        # not intersect with Q...
        R2x0 = min(right_indices)
        R2x1 = max(right_indices)
        R2y0 = round(Rget_y(R2x0)) # extrapolate segment for corresponding y
        R2y1 = round(Rget_y(R2x1)) # extrapolate segment for corresponding y
        R2_seg = ((R2x0, R2y0), (R2x1, R2y1))

        # Log new Q2/R1 seg and group
        max_i += 1
        all_new_segs.append(Q2_seg)
        update_dict(matches_dict, max_i, max_i+1)
        update_dict(matches_dict, max_i+1, max_i)
        max_i += 1
        all_new_segs.append(R1_seg)
        
        # Log new Q1 seg
        max_i += 1
        all_new_segs.append(Q1_seg)
        
        # log new R2 seg
        max_i += 1
        all_new_segs.append(R2_seg)

    if type_7:

        # Split Q into two patterns
        # that which intersects with
        # R....
        Q1x0 = min(overlap_indices)
        Q1x1 = max(overlap_indices)
        Q1y0 = round(Qget_y(Q1x0)) # extrapolate segment for corresponding y
        Q1y1 = round(Qget_y(Q1x1)) # extrapolate segment for corresponding y
        Q1_seg = ((Q1x0, Q1y0), (Q1x1, Q1y1))

        # And that part which does 
        # not intersect with Q...
        Q2x0 = min(right_indices)
        Q2x1 = max(right_indices)
        Q2y0 = round(Qget_y(Q2x0)) # extrapolate segment for corresponding y
        Q2y1 = round(Qget_y(Q2x1)) # extrapolate segment for corresponding y
        Q2_seg = ((Q2x0, Q2y0), (Q2x1, Q2y1))
        
        # Log new Q1 seg and group with R
        max_i += 1
        all_new_segs.append(Q1_seg)
        update_dict(matches_dict, j, max_i)
        update_dict(matches_dict, max_i, j)
        
        # Log new Q2 seg
        max_i += 1
        all_new_segs.append(Q2_seg)

    if type_8:

        # Split R into two patterns
        # that which does not intersects
        # with Q....
        R1x0 = min(left_indices)
        R1x1 = max(left_indices)
        R1y0 = round(Rget_y(R1x0)) # extrapolate segment for corresponding y
        R1y1 = round(Rget_y(R1x1)) # extrapolate segment for corresponding y
        R1_seg = ((R1x0, R1y0), (R1x1, R1y1))

        # And that part which does 
        # intersect with Q...
        R2x0 = min(overlap_indices)
        R2x1 = max(overlap_indices)
        R2y0 = round(Rget_y(R2x0)) # extrapolate segment for corresponding y
        R2y1 = round(Rget_y(R2x1)) # extrapolate segment for corresponding y
        R2_seg = ((R2x0, R2y0), (R2x1, R2y1))
        
        # Log new R2 seg and group with Q
        max_i += 1
        all_new_segs.append(R2_seg)
        update_dict(matches_dict, i, max_i)
        update_dict(matches_dict, max_i, i)
        
        # Log new R1 seg
        max_i += 1
        all_new_segs.append(R1_seg)

    if type_9:
        
        # Split Q into two patterns
        # that which does not intersect
        #  with R....
        Q1x0 = min(right_indices)
        Q1x1 = max(right_indices)
        Q1y0 = round(Qget_y(Q1x0)) # extrapolate segment for corresponding y
        Q1y1 = round(Qget_y(Q1x1)) # extrapolate segment for corresponding y
        Q1_seg = ((Q1x0, Q1y0), (Q1x1, Q1y1))

        # And that part which does 
        # intersect with R...
        Q2x0 = min(overlap_indices)
        Q2x1 = max(overlap_indices)
        Q2y0 = round(Qget_y(Q2x0)) # extrapolate segment for corresponding y
        Q2y1 = round(Qget_y(Q2x1)) # extrapolate segment for corresponding y
        Q2_seg = ((Q2x0, Q2y0), (Q2x1, Q2y1)) 

        # Split R into two patterns
        # that which intersects with
        # Q....
        R1x0 = min(overlap_indices)
        R1x1 = max(overlap_indices)
        R1y0 = round(Rget_y(R1x0)) # extrapolate segment for corresponding y
        R1y1 = round(Rget_y(R1x1)) # extrapolate segment for corresponding y
        R1_seg = ((R1x0, R1y0), (R1x1, R1y1))

        # And that part which does 
        # not intersect with Q...
        R2x0 = min(left_indices)
        R2x1 = max(left_indices)
        R2y0 = round(Rget_y(R2x0)) # extrapolate segment for corresponding y
        R2y1 = round(Rget_y(R2x1)) # extrapolate segment for corresponding y
        R2_seg = ((R2x0, R2y0), (R2x1, R2y1))

        # Log new R2/Q1 seg and group
        max_i += 1
        all_new_segs.append(R2_seg)
        update_dict(matches_dict, max_i, max_i+1)
        update_dict(matches_dict, max_i+1, max_i)
        max_i += 1
        all_new_segs.append(Q1_seg)
        
        # Log new R1 seg
        max_i += 1
        all_new_segs.append(R1_seg)
        
        # log new Q2 seg
        max_i += 1
        all_new_segs.append(Q2_seg)
    
    return all_new_segs, max_i, matches_dict


def update_dict(d, k, v):
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]


def get_dist(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5


def join_segments(segA, segB):
    ((Ax0,Ay0), (Ax1,Ay1)) = segA
    ((Bx0,By0), (Bx1,By1)) =  segB
    # which starts closer to origin?
    # use that ones start point and the others end point
    if get_dist((Ax0,Ay0), (0,0)) > get_dist((Bx0,By0), (0,0)):
        x0 = Bx0
        y0 = By0
        x1 = Ax1
        y1 = Ay1
    else:
        x0 = Ax0
        y0 = Ay0
        x1 = Bx1
        y1 = By1
    return ((x0,y0), (x1, y1))


def join_all_segments(all_segments, min_diff_trav_seq):
    group_join_dict = {}
    for i, ((Qx0, Qy0), (Qx1, Qy1)) in tqdm.tqdm(list(enumerate(all_segments))):
        for j, [(Rx0, Ry0), (Rx1, Ry1)] in enumerate(all_segments):
            if i == j:
                continue

            if abs(Rx0-Qx1) > min_diff_trav_seq:
                continue
            if abs(Ry0-Qy1) > min_diff_trav_seq:
                continue

            # if distance between start an end
            if get_dist((Rx0, Ry0), (Qx1, Qy1)) < min_diff_trav_seq:
                update_dict(group_join_dict, i, j)
                update_dict(group_join_dict, j, i)
                continue
            # if distance between end and start
            elif get_dist((Rx1, Ry1), (Qx0, Qy0)) < min_diff_trav_seq:
                update_dict(group_join_dict, i, j)
                update_dict(group_join_dict, j, i)
                continue

    all_prox_groups = matches_dict_to_groups(group_join_dict)
    to_skip = []
    all_segments_joined = []
    for group in all_prox_groups:
        to_skip += group
        seg = all_segments[group[0]]
        for g in group[1:]:
            seg2 = all_segments[g]
            seg = join_segments(seg, seg2)
        all_segments_joined.append(seg)
    all_segments_joined += [x for i,x in enumerate(all_segments) if i not in to_skip]
    return all_segments_joined


def learn_relationships_and_break(
    i, j, Qx0, Qy0, Qx1, Qy1, Rx0, Ry0, Rx1, Ry1, min_length_cqt, 
    new_segments, contains_dict, is_subset_dict, shares_common):

    # functions that define line through query(Q) segment
    Qget_x, Qget_y = line_through_points(Qx0, Qy0, Qx1, Qy1)
    # get indices corresponding to query(Q)
    Q_indices = set(range(Qx0, Qx1+1))

    # functions that define line through returned(R) segment
    Rget_x, Rget_y = line_through_points(Rx0, Ry0, Rx1, Ry1)
    # get indices corresponding to query(Q)
    R_indices = set(range(Rx0, Rx1+1))


    # query on the left
    if Qx0 <= Rx0:
        # indices in common between query(Q) and returned(R)
        left_indices = Q_indices.difference(R_indices)
        overlap_indices = Q_indices.intersection(R_indices)
        right_indices = R_indices.difference(Q_indices)

        # which parts in the venn diagram
        # betweem Q and R are large enough to
        # be considered
        left_sig = len(left_indices) >= min_length_cqt
        overlap_sig = len(overlap_indices) >= min_length_cqt
        right_sig = len(right_indices) >= min_length_cqt

        # which type of match (if any). 
        # See above for explanation
        type_1 = not overlap_indices
        type_2 = not overlap_sig and overlap_indices
        type_3 = all([overlap_sig, not left_sig, not right_sig])

        type_4 = all([not left_sig, overlap_sig, right_sig])
        type_5 = all([left_sig, overlap_sig, not right_sig])
        type_6 = all([left_sig, overlap_sig, right_sig])

        type_7 = False
        type_8 = False
        type_9 = False

    # query on the right
    else:
        # indices in common between query(Q) and returned(R)
        left_indices = R_indices.difference(Q_indices)
        overlap_indices = R_indices.intersection(Q_indices)
        right_indices = Q_indices.difference(R_indices)

        # which parts in the venn diagram
        # betweem Q and R are large enough to
        # be considered
        left_sig = len(left_indices) >= min_length_cqt
        overlap_sig = len(overlap_indices) >= min_length_cqt
        right_sig = len(right_indices) >= min_length_cqt

        # which type of match (if any). 
        # See above for explanation
        type_1 = not overlap_indices
        type_2 = not overlap_sig and overlap_indices
        type_3 = all([overlap_sig, not left_sig, not right_sig])

        type_4 = False
        type_5 = False
        type_6 = False

        type_7 = all([not left_sig, overlap_sig, right_sig])
        type_8 = all([left_sig, overlap_sig, not right_sig])
        type_9 = all([left_sig, overlap_sig, right_sig])

    ###########################
    ### Create New Segments ###
    ###########################
    if type_4 or type_7:
        x0 = round(min(overlap_indices))
        x1 = round(max(overlap_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        overlap_seg = ((x0,y0),(x1,y1))
        new_segments.append(overlap_seg)
        
        # index of new segment
        Oi = len(new_segments) - 1

        x0 = round(min(right_indices))
        x1 = round(max(right_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        right_seg = ((x0,y0),(x1,y1))
        new_segments.append(right_seg)

        # index of new segment
        Ri = len(new_segments) - 1

    if type_5 or type_8:
        x0 = round(min(overlap_indices))
        x1 = round(max(overlap_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        overlap_seg = ((x0,y0),(x1,y1))
        new_segments.append(overlap_seg)

        # index of new segment
        Oi = len(new_segments) - 1

        x0 = round(min(left_indices))
        x1 = round(max(left_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        left_seg = ((x0,y0),(x1,y1))
        new_segments.append(left_seg)

        # index of new segment
        Li = len(new_segments) - 1

    if type_6 or type_9:
        x0 = round(min(overlap_indices))
        x1 = round(max(overlap_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        overlap_seg = ((x0,y0),(x1,y1))
        new_segments.append(overlap_seg)

        # index of new segment
        Oi = len(new_segments) - 1

        x0 = round(min(left_indices))
        x1 = round(max(left_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        left_seg = ((x0,y0),(x1,y1))
        new_segments.append(left_seg)

        # index of new segment
        Li = len(new_segments) - 1

        x0 = round(min(right_indices))
        x1 = round(max(right_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        right_seg = ((x0,y0),(x1,y1))
        new_segments.append(right_seg)  

        # index of new segment
        Ri = len(new_segments) - 1

    ############################
    ### Record Relationships ###
    ############################
    if type_4:
        update_dict(contains_dict, j, i)
        update_dict(contains_dict, j, Oi)
        update_dict(contains_dict, j, Ri)

        update_dict(is_subset_dict, i, j)
        update_dict(is_subset_dict, Oi, j)
        update_dict(is_subset_dict, Ri, j)

    if type_5:
        update_dict(contains_dict, i, j)
        update_dict(contains_dict, i, Oi)
        update_dict(contains_dict, i, Li)

        update_dict(is_subset_dict, j, i)
        update_dict(is_subset_dict, Oi, i)
        update_dict(is_subset_dict, Li, i)

    if type_6:
        update_dict(contains_dict, i, Oi)
        update_dict(contains_dict, j, Oi)
        update_dict(contains_dict, i, Li)
        update_dict(contains_dict, j, Ri)

        update_dict(is_subset_dict, Oi, i)
        update_dict(is_subset_dict, Oi, j)
        update_dict(is_subset_dict, Li, i)
        update_dict(is_subset_dict, Ri, j)

        update_dict(shares_common, i, j)
        update_dict(shares_common, j, i)

    if type_7:
        update_dict(contains_dict, j, i)
        update_dict(contains_dict, j, Oi)
        update_dict(contains_dict, j, Ri)

        update_dict(is_subset_dict, i, j)
        update_dict(is_subset_dict, Oi, j)
        update_dict(is_subset_dict, Ri, j)

    if type_8:
        update_dict(contains_dict, j, j)
        update_dict(contains_dict, j, Oi)
        update_dict(contains_dict, j, Li)

        update_dict(is_subset_dict, j, j)
        update_dict(is_subset_dict, Oi, j)
        update_dict(is_subset_dict, Li, j)

    if type_9:
        update_dict(contains_dict, i, Oi)
        update_dict(contains_dict, j, Oi)
        update_dict(contains_dict, j, Li)
        update_dict(contains_dict, i, Ri)

        update_dict(is_subset_dict, Oi, i)
        update_dict(is_subset_dict, Oi, j)
        update_dict(is_subset_dict, Li, j)
        update_dict(is_subset_dict, Ri, i)

        update_dict(shares_common, i, j)
        update_dict(shares_common, j, i)

    return new_segments, shares_common, is_subset_dict, contains_dict


def identify_matches(i, j, Qx0, Qy0, Qx1, Qy1, Rx0, Ry0, Rx1, Ry1, min_length_cqt, matches_dict):

    # get indices corresponding to query(Q)
    Q_indices = set(range(Qx0, Qx1+1))

    # get indices corresponding to query(Q)
    R_indices = set(range(Rx0, Rx1+1))

    # query on the left
    if Qx0 <= Rx0:
        # indices in common between query(Q) and returned(R)
        left_indices = Q_indices.difference(R_indices)
        overlap_indices = Q_indices.intersection(R_indices)
        right_indices = R_indices.difference(Q_indices)

        # which parts in the venn diagram
        # betweem Q and R are large enough to
        # be considered
        left_sig = len(left_indices) >= min_length_cqt
        overlap_sig = len(overlap_indices) >= min_length_cqt
        right_sig = len(right_indices) >= min_length_cqt

    # query on the right
    else:
        # indices in common between query(Q) and returned(R)
        left_indices = R_indices.difference(Q_indices)
        overlap_indices = R_indices.intersection(Q_indices)
        right_indices = Q_indices.difference(R_indices)

        # which parts in the venn diagram
        # betweem Q and R are large enough to
        # be considered
        left_sig = len(left_indices) >= min_length_cqt
        overlap_sig = len(overlap_indices) >= min_length_cqt
        right_sig = len(right_indices) >= min_length_cqt

    if all([overlap_sig, not left_sig, not right_sig]):
        update_dict(matches_dict, i, j)
        update_dict(matches_dict, j, i)

    return matches_dict




















