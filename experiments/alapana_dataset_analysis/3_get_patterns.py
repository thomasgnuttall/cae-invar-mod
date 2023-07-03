############# NEW MODEL #############
from experiments.alapana_dataset_analysis.main import main

import faulthandler


run_name = 'icmpc_old_time'

faulthandler.enable()
sr = 44100
cqt_window = 1984
s1 = None
s2 = None
gap_interp = 0.35
stab_hop_secs = 0.2
min_stability_length_secs = 1.0
freq_var_thresh_stab = 60
conv_filter_str = 'sobel'
gauss_sigma = None
cont_thresh = 0.15
etc_kernel_size = 10
binop_dim = 3
min_diff_trav = 0.5 #0.1
min_in_group = 2
match_tol = 1
ext_mask_tol = 0.5
n_dtw = 10
thresh_cos = None
top_n = 1000
write_plots = True
write_audio = True
write_patterns = True
write_annotations = False
partial_perc = 0.66
perc_tail = 0.5
plot = False
min_pattern_length_seconds = 1.5

group_len_var = 1 # Current Best: 1
thresh_dtw = 4.5 # Current Best: 8
dupl_perc_overlap_intra = 0.6 # Current Best: 0.6
dupl_perc_overlap_inter = 0.8 # Current Best: 0.75


track_names = [
    "2018_11_13_am_Sec_1_P2_Varaali_slates",
    "2018_11_13_am_Sec_2_P1_Shankara_slates",
    "2018_11_13_am_Sec_3_P1_Anandabhairavi_slates",
    "2018_11_13_am_Sec_3_P3_Kalyani_slates",
    "2018_11_13_am_Sec_3_P5_Todi_slates",
    "2018_11_13_am_Sec_3_P8_Bilahari_B_slates",
    "2018_11_13_am_Sec_4_P1_Atana_slatesB",
    "2018_11_13_pm_Sec_1_P1_Varaali_slates",
    "2018_11_13_pm_Sec_1_P2_Anandabhairavi_slates",
    "2018_11_13_pm_Sec_2_P1_Kalyani_slates",
    "2018_11_13_pm_Sec_3_P1_Todi_A_slates",
    "2018_11_13_pm_Sec_3_P2_Todi_B_slates",
    "2018_11_13_pm_Sec_4_P1_Shankara_slates",
    "2018_11_13_pm_Sec_5_P1_Bilahari_slates",
    "2018_11_13_pm_Sec_5_P2_Atana_slates",
    "2018_11_15_Sec_10_P1_Atana_slates",
    "2018_11_15_Sec_12_P1_Kalyani_slates",
    "2018_11_15_Sec_13_P1_Bhairavi_slates",
    "2018_11_15_Sec_1_P1_Anandabhairavi_A_slates",
    "2018_11_15_Sec_2_P1_Anandabhairavi_B_slates",
    "2018_11_15_Sec_3_P1_Anandabhairavi_C_slates",
    "2018_11_15_Sec_4_P1_Shankara_slates",
    "2018_11_15_Sec_6_P1_Varaali_slates",
    "2018_11_15_Sec_8_P1_Bilahari_slates",
    "2018_11_15_Sec_9_P1_Todi_slates",
    "2018_11_18_am_Sec_1_P1_Varaali_slates",
    "2018_11_18_am_Sec_2_P2_Shankara_slates",
    "2018_11_18_am_Sec_3_P1_Anandabhairavi_A_slates",
    "2018_11_18_am_Sec_3_P2_Anandabhairavi_B_slates",
    "2018_11_18_am_Sec_4_P1_Kalyani_slates",
    "2018_11_18_am_Sec_5_P1_Bhairavi_slates",
    "2018_11_18_am_Sec_5_P3_Bilahari_slates",
    "2018_11_18_am_Sec_6_P1_Atana_A_slates",
    "2018_11_18_am_Sec_6_P2_Atana_B_slates",
    "2018_11_18_am_Sec_6_P4_Todi_slates",
    "2018_11_18_am_Sec_6_P6_Bilahari_B",
    "2018_11_18_pm_Sec_1_P1_Shankara_slates",
    "2018_11_18_pm_Sec_1_P2_Varaali_slates",
    "2018_11_18_pm_Sec_1_P3_Bilahari_slates",
    "2018_11_18_pm_Sec_2_P2_Anandabhairavi_full_slates",
    "2018_11_18_pm_Sec_3_P1_Kalyani_slates",
    "2018_11_18_pm_Sec_4_P1_Bhairavi_slates",
    "2018_11_18_pm_Sec_4_P2_Atana_slates",
    "2018_11_18_pm_Sec_5_P1_Todi_full_slates",
    "2018_11_18_pm_Sec_5_P2_Sahana_slates"
]

failed = []
total = len(track_names)
i = 0
for t in track_names:
    # each date corresponds to performer
    # each performer has a different bin_thresh
    if t == "2018_11_13_am_Sec_1_P2_Varaali_slates":
        bin_thresh = 0.085
    if t == "2018_11_13_am_Sec_2_P1_Shankara_slates":
        bin_thresh = 0.035
    if t == "2018_11_13_am_Sec_3_P1_Anandabhairavi_slates":
        bin_thresh = 0.065
    if t == "2018_11_13_am_Sec_3_P3_Kalyani_slates":
        bin_thresh = 0.015
    if t == "2018_11_13_am_Sec_3_P5_Todi_slates":
        bin_thresh = 0.065
    if t == "2018_11_13_am_Sec_3_P8_Bilahari_B_slates":
        bin_thresh = 0.045
    if t == "2018_11_13_am_Sec_4_P1_Atana_slatesB":
        bin_thresh = 0.045
    if t == "2018_11_13_pm_Sec_1_P1_Varaali_slates":
        bin_thresh = 0.065
    if t == "2018_11_13_pm_Sec_1_P2_Anandabhairavi_slates":
        bin_thresh = 0.01
    if t == "2018_11_13_pm_Sec_2_P1_Kalyani_slates":
        bin_thresh = 0.035
    if t == "2018_11_13_pm_Sec_3_P1_Todi_A_slates":
        bin_thresh = 0.06
    if t == "2018_11_13_pm_Sec_3_P2_Todi_B_slates":
        bin_thresh = 0.01
    if t == "2018_11_13_pm_Sec_4_P1_Shankara_slates":
        bin_thresh = 0.02
    if t == "2018_11_13_pm_Sec_5_P1_Bilahari_slates":
        bin_thresh = 0.1
    if t == "2018_11_13_pm_Sec_5_P2_Atana_slates":
        bin_thresh = 0.01
    if t == "2018_11_15_Sec_10_P1_Atana_slates":
        bin_thresh = 0.015
    if t == "2018_11_15_Sec_12_P1_Kalyani_slates":
        bin_thresh = 0.01
    if t == "2018_11_15_Sec_13_P1_Bhairavi_slates":
        bin_thresh = 0.03
    if t == "2018_11_15_Sec_1_P1_Anandabhairavi_A_slates":
        bin_thresh = 0.01
    if t == "2018_11_15_Sec_2_P1_Anandabhairavi_B_slates":
        bin_thresh = 0.01
    if t == "2018_11_15_Sec_3_P1_Anandabhairavi_C_slates":
        bin_thresh = 0.01
    if t == "2018_11_15_Sec_4_P1_Shankara_slates":
        bin_thresh = 0.01
    if t == "2018_11_15_Sec_6_P1_Varaali_slates":
        bin_thresh = 0.065
    if t == "2018_11_15_Sec_8_P1_Bilahari_slates":
        bin_thresh = 0.015
    if t == "2018_11_15_Sec_9_P1_Todi_slates":
        bin_thresh = 0.015
    if t == "2018_11_18_am_Sec_1_P1_Varaali_slates":
        bin_thresh = 0.09
    if t == "2018_11_18_am_Sec_2_P2_Shankara_slates":
        bin_thresh = 0.08
    if t == "2018_11_18_am_Sec_3_P1_Anandabhairavi_A_slates":
        bin_thresh = 0.045
    if t == "2018_11_18_am_Sec_3_P2_Anandabhairavi_B_slates":
        bin_thresh = 0.065
    if t == "2018_11_18_am_Sec_4_P1_Kalyani_slates":
        bin_thresh = 0.095
    if t == "2018_11_18_am_Sec_5_P1_Bhairavi_slates":
        bin_thresh = 0.035
    if t == "2018_11_18_am_Sec_5_P3_Bilahari_slates":
        bin_thresh = 0.065
    if t == "2018_11_18_am_Sec_6_P1_Atana_A_slates":
        bin_thresh = 0.01
    if t == "2018_11_18_am_Sec_6_P2_Atana_B_slates":
        bin_thresh = 0.025
    if t == "2018_11_18_am_Sec_6_P4_Todi_slates":
        bin_thresh = 0.09
    if t == "2018_11_18_am_Sec_6_P6_Bilahari_B":
        bin_thresh = 0.065
    if t == "2018_11_18_pm_Sec_1_P1_Shankara_slates":
        bin_thresh = 0.01
    if t == "2018_11_18_pm_Sec_1_P2_Varaali_slates":
        bin_thresh = 0.01
    if t == "2018_11_18_pm_Sec_1_P3_Bilahari_slates":
        bin_thresh = 0.01
    if t == "2018_11_18_pm_Sec_2_P2_Anandabhairavi_full_slates":
        bin_thresh = 0.01
    if t == "2018_11_18_pm_Sec_3_P1_Kalyani_slates":
        bin_thresh = 0.01
    if t == "2018_11_18_pm_Sec_4_P1_Bhairavi_slates":
        bin_thresh = 0.01
    if t == "2018_11_18_pm_Sec_4_P2_Atana_slates":
        bin_thresh = 0.01
    if t == "2018_11_18_pm_Sec_5_P1_Todi_full_slates":
        bin_thresh = 0.01
    if t == "2018_11_18_pm_Sec_5_P2_Sahana_slates":
        bin_thresh = 0.015

    i += 1
 #   try:
    title = f'{i}/{total} | Track name: {t}, bin_thresh: {bin_thresh}'
    print(title)
    print('-'*len(title))
    bin_thresh_segment = bin_thresh*0.75
    main(
        t, run_name, sr, cqt_window, s1, s2,
        gap_interp, stab_hop_secs, min_stability_length_secs,
        60, conv_filter_str, bin_thresh,
        bin_thresh_segment, perc_tail,  gauss_sigma, cont_thresh,
        etc_kernel_size, binop_dim, min_diff_trav, min_pattern_length_seconds,
        min_in_group, match_tol, ext_mask_tol, n_dtw, thresh_dtw, thresh_cos, group_len_var, 
        dupl_perc_overlap_inter, dupl_perc_overlap_intra, None,
        None, partial_perc, top_n, True,
        write_audio, write_patterns, False, plot=False)
#    except Exception as e:
        #failed.append((t,e))
        #print('FAILED')
        #print(e)



############# Transfer to final Folder ###############
######################################################
from distutils.dir_util import copy_tree
from exploration.io import create_if_not_exists

for t in track_names:
    print(f'transferring {t}')
    folder1 = f'/Volumes/Shruti/asplab2/cae-invar/data/self_similarity/{t}/results/{run_name}/'

    folder2 = f'/Volumes/Shruti/FOR_LARA/{run_name}/{t}/results/{run_name}/'
    create_if_not_exists(folder2)
    copy_tree(folder1, folder2)


#####################################
############ For Lara ###############
#####################################
from exploration.io import load_pkl
import pandas as pd
out_dir = f'/Volumes/Shruti/FOR_LARA/{run_name}/'
failed_tracks = []
for track_name in track_names:
    starts  = load_pkl(f'{out_dir}/{track_name}/results/{run_name}/starts.pkl')
    lengths = load_pkl(f'{out_dir}/{track_name}/results/{run_name}/lengths.pkl')

    df = pd.DataFrame(columns=['start','end','group'])

    timestep = 0.010000958497076582
    for i,group in enumerate(starts):
        for j,s in enumerate(group):
            l = lengths[i][j]
            s1 = s*timestep
            s2 = (l+s)*timestep
            df = df.append({
                'start':s1,
                'end':s2,
                'group':i,
                'occurrence':j
                }, ignore_index=True)

    df.to_csv(f'{out_dir}/{track_name}/results/{run_name}/groups.csv', index=False)


