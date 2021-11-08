import numpy as np

def contains_silence(seq, thresh=0.05):
    """If more than <thresh> of <seq> is 0, return True"""
    return sum(seq==0)/len(seq) > thresh


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def too_stable(seq, dev_thresh=5, perc_thresh=0.63, window=200):
    """If a sufficient proportion of <seq> is "stable" return True"""
    if window > len(seq):
        window=len(seq)
    mu_ = seq[:window-1]
    mu = np.concatenate([mu_, moving_average(seq, window)])

    dev_arr = abs(mu-seq)
    dev_seq = dev_arr[np.where(~np.isnan(dev_arr))]
    bel_thresh = dev_seq < dev_thresh

    perc = np.count_nonzero(bel_thresh)/len(dev_seq)

    if perc >= perc_thresh:
        is_stable = 1
    else:
        is_stable = 0
    
    return is_stable


def start_with_silence(seq):
    return any([seq[0] == 0, all(seq[:100]==0)])


def min_gap(seq, length=86):
    seq2 = np.trim_zeros(seq)
    m1 = np.r_[False, seq2==0, False]
    idx = np.flatnonzero(m1[:-1] != m1[1:])
    if len(idx) > 0:
        out = (idx[1::2]-idx[::2])
        if any(o >= length for o in out):
            return True
    return False


def is_stable(seq, max_var):
    mu = np.nanmean(seq)
    maximum = np.nanmax(seq)
    minimum = np.nanmin(seq)
    if (maximum < mu + max_var) and (minimum > mu - max_var):
        return 1
    else:
        return 0

def reduce_stability_mask(stable_mask, min_stability_length_secs, timestep):
    min_stability_length = int(min_stability_length_secs/timestep)
    num_one = 0
    indices = []
    for i,s in enumerate(stable_mask):
        if s == 1:
            num_one += 1
            indices.append(i)
        else:
            if num_one < min_stability_length:
                for ix in indices:
                    stable_mask[ix] = 0
            num_one = 0
            indices = []
    return stable_mask


def add_center_to_mask(stable_mask):
    num_one = 0
    indices = []
    for i,s in enumerate(stable_mask):
        if s == 1:
            num_one += 1
            indices.append(i)
        else:
            li = len(indices)
            if li:
                middle = indices[int(len(indices)/2)]
                stable_mask[middle] = 2
                num_one = 0
                indices = []
    return stable_mask


def get_stability_mask(raw_pitch, min_stability_length_secs, stability_hop_secs, var_thresh, timestep):
    stab_hop = int(stability_hop_secs/timestep)
    reverse_raw_pitch = np.flip(raw_pitch)

    # apply in both directions to array to account for hop_size errors
    stable_mask_1 = [is_stable(raw_pitch[s:s+stab_hop], var_thresh) for s in range(len(raw_pitch))]
    stable_mask_2 = [is_stable(reverse_raw_pitch[s:s+stab_hop], var_thresh) for s in range(len(reverse_raw_pitch))]
    
    silence_mask = raw_pitch == 0

    zipped = zip(stable_mask_1, np.flip(stable_mask_2), silence_mask)
    
    stable_mask = np.array([int((any([s1,s2]) and not sil)) for s1,s2,sil in zipped])

    stable_mask = reduce_stability_mask(stable_mask, min_stability_length_secs, timestep)

    stable_mask = add_center_to_mask(stable_mask)

    return stable_mask