from collections import Counter

import textgrid
import numpy as np
import pandas as pd

import math

def load_annotations_brindha(annotations_path, min_m=None, max_m=None):

    annotations_orig = pd.read_csv(annotations_path, sep='\t')
    annotations_orig.columns = ['tier', 'not_used', 's1', 's2', 'duration', 'text']
    
    annotations_orig['s1'] = pd.to_datetime(annotations_orig['s1']).apply(lambda y: y.time())
    annotations_orig['s2'] = pd.to_datetime(annotations_orig['s2']).apply(lambda y: y.time())
    annotations_orig['duration'] = pd.to_datetime(annotations_orig['duration']).apply(lambda y: y.time())

    annotations_orig['s1'] = annotations_orig['s1'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)
    annotations_orig['s2'] = annotations_orig['s2'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)
    annotations_orig['duration'] = annotations_orig['duration'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)

    if min_m:
        annotations_orig = annotations_orig[annotations_orig['duration'].astype(float)>min_m]
    if max_m:
        annotations_orig = annotations_orig[annotations_orig['duration'].astype(float)<max_m]

    annotations_orig = annotations_orig[annotations_orig['tier'].isin(['underlying_full_phrase','underlying_sancara'])]
    good_text = [k for k,v in Counter(annotations_orig['text']).items() if v>1]
    annotations_orig = annotations_orig[annotations_orig['text'].isin(good_text)]

    annotations_orig = annotations_orig[annotations_orig['s2']- annotations_orig['s1']>=1]
    
    return annotations_orig[['tier', 's1', 's2', 'text']]



def get_coverage(pitch, starts_seq_exc, lengths_seq_exc):
    pitch_coverage = pitch.copy()
    pitch_coverage[:] = 0

    for i, group in enumerate(starts_seq_exc):
        for j, s in enumerate(group):
            l = lengths_seq_exc[i][j]
            pitch_coverage[s:s+l] = 1

    return np.sum(pitch_coverage)/len(pitch_coverage)

def is_match_v2(sp, lp, sa, ea, partial_perc=0.3):

    ep = sp + lp
    
    # partial if identified pattern captures a
    # least <partial_perc> of annotation
    la = (ea-sa) # length of annotation

    overlap = 0

    # pattern starts in annotation
    if (sa <= sp <= ea):
        if ep < ea:
            overlap = (ep-sp)
        else:
            overlap = (ea-sp)

    # pattern ends in annotation
    if (sa <= ep <= ea):
        if sa < sp:
            overlap = (ep-sp)
        else:
            overlap = (ep-sa)

    # if intersection between annotation and returned pattern is 
    # >= <partial_perc> of each its a match!
    if overlap/la >= partial_perc and overlap/lp >= partial_perc:
        return 'match'
    else:
        return None


def evaluate_annotations(annotations_raw, starts, lengths, partial_perc):
    annotations = annotations_raw.copy()
    results_dict = {}
    group_num_dict = {}
    occ_num_dict = {}
    is_matched_arr = []
    for i, seq_group in enumerate(starts):
        length_group  = lengths[i]
        ima = []
        for j, seq in enumerate(seq_group):
            im = 0
            length = length_group[j]
            for ai, (tier, s1, s2, text) in zip(annotations.index, annotations.values):
                matched = is_match_v2(seq, length, s1, s2, partial_perc=partial_perc)
                if matched:
                    im = 1
                    if ai not in results_dict:
                        results_dict[ai] = matched
                        group_num_dict[ai] = i
                        occ_num_dict[ai] = j
            ima = ima + [im]
        is_matched_arr.append(ima)

    annotations['match']     = [results_dict[i] if i in results_dict else 'no match' for i in annotations.index]
    annotations['group_num'] = [group_num_dict[i] if i in group_num_dict else None for i in annotations.index]
    annotations['occ_num']   = [occ_num_dict[i] if i in occ_num_dict else None for i in annotations.index]

    return annotations, is_matched_arr


def evaluate(annotations_raw, starts, lengths, partial_perc):
    annotations, is_matched = evaluate_annotations(annotations_raw, starts, lengths, partial_perc)
    ime = [x for y in is_matched for x in y]
    precision = sum(ime)/len(ime) if ime else 1
    recall = sum(annotations['match']!='no match')/len(annotations)
    f1 = f1_score(precision, recall)
    return recall, precision, f1, annotations


def f1_score(p,r):
    return 2*p*r/(p+r) if (p+r != 0) else 0