from collections import Counter

import textgrid
import numpy as np
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


def get_metrics(annotations, starts_sec_exc, suffix=''):
    
    if not starts_sec_exc:
        return {
            f'n_groups': 0,
            f'n_patterns': 0,
            f'max_n_in_group': np.nan,
            f'partial_match_precision{suffix}': np.nan,
            f'partial_match_recall{suffix}': np.nan,
            f'full_match_precision{suffix}': np.nan,
            f'full_match_recall{suffix}': np.nan,
            f'partial_group_precision{suffix}': np.nan,
            f'full_group_precision{suffix}': np.nan,
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
        f'n_groups': n_groups,
        f'n_patterns': n_patterns,
        f'max_n_in_group': max_n_in_group,
        f'partial_match_precision{suffix}': partial_prec,
        f'partial_match_recall{suffix}': partial_rec,
        f'full_match_precision{suffix}': full_prec,
        f'full_match_recall{suffix}': full_rec,
        f'partial_group_precision{suffix}': group_partial_prec,
        f'full_group_precision{suffix}': group_full_prec,
    }


def load_annotations_new(annotations_path):

    annotations_orig = pd.read_csv(annotations_path, sep='\t')
    annotations_orig.columns = ['s1', 's2', 'duration', 'short_motif', 'motif', 'phrase']
    for c in ['short_motif', 'motif', 'phrase']:
        l1 = len(annotations_orig)

        # ensure_dir that whitespace is
        annotations_orig[c] = annotations_orig[c].apply(lambda y: y.strip() if pd.notnull(y) else np.nan)   

        # remove phrases that occur once
        one_occ = [x for x,y in Counter(annotations_orig[c].values).items() if y == 1]
        annotations_orig = annotations_orig[~annotations_orig[c].isin(one_occ)]
        l2 = len(annotations_orig)
        print(f'    - {l1-l2} {c}s removed from annotations for only occurring once')

    annotations_orig['s1'] = pd.to_datetime(annotations_orig['s1']).apply(lambda y: y.time())
    annotations_orig['s2'] = pd.to_datetime(annotations_orig['s2']).apply(lambda y: y.time())

    annotations_orig['s1'] = annotations_orig['s1'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)
    annotations_orig['s2'] = annotations_orig['s2'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)

    annotations_orig['tier'] = annotations_orig[['short_motif', 'motif', 'phrase']].apply(pd.Series.first_valid_index, axis=1)
    annotations_orig['text'] = [annotations_orig.loc[k, v] if v is not None else None for k, v in annotations_orig['tier'].iteritems()]

    annotations_orig = annotations_orig[['tier', 's1', 's2', 'text']]

    return annotations_orig


def evaluate_all_tiers(annotations_orig, starts_sec_exc, lengths_sec_exc, eval_tol):
    annotations_short_motif = annotations_orig[annotations_orig['tier']=='short_motif']
    annotations_motif = annotations_orig[annotations_orig['tier']=='motif']
    annotations_phrase = annotations_orig[annotations_orig['tier']=='phrase']

    annotations_orig = evaluate_annotations(annotations_orig, starts_sec_exc, lengths_sec_exc, eval_tol)
    metrics_orig = get_metrics(annotations_orig, starts_sec_exc, '_all')

    annotations_short_motif = evaluate_annotations(annotations_short_motif, starts_sec_exc, lengths_sec_exc, eval_tol)
    metrics_short_motif = get_metrics(annotations_short_motif, starts_sec_exc, '_short_motif')

    annotations_motif = evaluate_annotations(annotations_motif, starts_sec_exc, lengths_sec_exc, eval_tol)
    metrics_motif = get_metrics(annotations_motif, starts_sec_exc, '_motif')

    annotations_phrase = evaluate_annotations(annotations_phrase, starts_sec_exc, lengths_sec_exc, eval_tol)
    metrics_phrase = get_metrics(annotations_phrase, starts_sec_exc, '_phrase')

    all_metrics = {}
    all_metrics.update(metrics_orig)
    all_metrics.update(metrics_short_motif)
    all_metrics.update(metrics_motif)
    all_metrics.update(metrics_phrase)

    return all_metrics