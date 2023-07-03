from experiments.alapana_dataset_analysis.main import main
track_names = [
    '2018_11_13_am_Sec_3_P1_Anandabhairavi_V.2',
    '2018_11_13_pm_Sec_3_P1_Todi_A',
    '2018_11_18_am_Sec_6_P6_Bilahari_B',
    '2018_11_19_Sec2_P1_Kalyani',
    '2018_11_18_pm_Sec5_P2_Sahana',
    '2018_11_15_Sec_4_P1_Shankara',
    '2018_11_13_am_Sec_1_P2_Varaali_V.2',
    '2018_11_13_pm_Sec_5_P2_Atana',
    '2018_11_18_am_Sec_3_P2_Anandabhairavi_B',
    '2018_11_18_pm_Sec4_P2_Atana',
    '2018_11_13_am_Sec_2_P1_Shankara_V.2',
    '2018_11_15_Sec_10_P1_Atana',
    '2018_11_18_am_Sec_4_P1_Kalyani',   
    '2018_11_18_pm_Sec5_P1_Todi_full',
    '2018_11_15_Sec_12_P1_Kalyani',
    '2018_11_18_am_Sec_5_P1_Bhairavi',
    '2018_11_13_am_Sec_3_P3_Kalyani_V.2',
    '2018_11_15_Sec_13_P1_Bhairavi',
    '2018_11_18_am_Sec_5_P3_Bilahari',
    '2018_11_19_Sec1_P1_Anandabhairavi_A_unfinished',
    '2018_11_13_am_Sec_3_P5_Todi_V.2',
    '2018_11_15_Sec_1_P1_Anandabhairavi_A',
    '2018_11_18_am_Sec_6_P1_Atana_A',
    '2018_11_19_Sec1_P2_Anandabhairavi_B_full',
    '2018_11_13_am_Sec_3_P8_Bilahari_B_V.2',
    '2018_11_15_Sec_2_P1_Anandabhairavi_B',
    '2018_11_18_am_Sec_6_P2_Atana_B',
    '2018_11_19_Sec1_P3_Varaali',
    '2018_11_13_am_Sec_4_P1_Atana_V.2',
    '2018_11_15_Sec_3_P1_Anandabhairavi_C',
    '2018_11_18_am_Sec_6_P4_Todi',
    '2018_11_13_pm_Sec_1_P1_Varaali',
    '2018_11_19_Sec2_P3_Todi',
    '2018_11_13_pm_Sec_1_P2_Anandabhairavi',
    '2018_11_15_Sec_6_P1_Varaali',
    '2018_11_18_pm_Sec1_P1_Shankara',
    '2018_11_19_Sec3_P1_Bilahari',
    '2018_11_13_pm_Sec_2_P1_Kalyani',
    '2018_11_15_Sec_8_P1_Bilahari',
    '2018_11_18_pm_Sec1_P2_Varaali',
    '2018_11_19_Sec3_P3_Bhairavi',
    '2018_11_15_Sec_9_P1_Todi',
    '2018_11_18_pm_Sec1_P3_Bilahari',
    '2018_11_19_Sec4_P1_Shankara',
    '2018_11_13_pm_Sec_3_P2_Todi_B',
    '2018_11_18_am_Sec_1_P1_Varaali',
    '2018_11_18_pm_Sec2_P2_Anandabhairavi_full',
    '2018_11_19_Sec4_P3_Atana',
    '2018_11_13_pm_Sec_4_P1_Shankara',
    '2018_11_18_am_Sec_2_P2_Shankara',
    '2018_11_18_pm_Sec3_P1_Kalyani',
    '2018_11_13_pm_Sec_5_P1_Bilahari',
    '2018_11_18_am_Sec_3_P1_Anandabhairavi_A',
    '2018_11_18_pm_Sec4_P1_Bhairavi'
]
import faulthandler

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
plot=False
min_pattern_length_seconds = 1.5

group_len_var = 1 # Current Best: 1
thresh_dtw = 4.5 # Current Best: 8
dupl_perc_overlap_intra = 0.6 # Current Best: 0.6
dupl_perc_overlap_inter = 0.8 # Current Best: 0.75

i=0
track_name = '2018_11_13_am_Sec_3_P1_Anandabhairavi_V.2'
run_name='test'
for dupl_perc_overlap_inter in [0.75]:
    for dupl_perc_overlap_intra in [0.6]:
        for group_len_var in [1]:
            for bin_thresh in [0.005]:
                if dupl_perc_overlap_intra > dupl_perc_overlap_inter:
                    continue
                for track_name in ['2018_11_13_am_Sec_3_P1_Anandabhairavi_V.2', '2018_11_13_pm_Sec_3_P1_Todi_A', '2018_11_15_Sec_6_P1_Varaali', '2018_11_18_am_Sec_6_P6_Bilahari_B', '2018_11_19_Sec2_P1_Kalyani', '2018_11_18_pm_Sec1_P3_Bilahari']:
                    i+=1
                    run_name = f"bin_thresh={bin_thresh}" #_group_len_var={group_len_var}_dintra={dupl_perc_overlap_intra}_dinter={dupl_perc_overlap_inter}"
                    print(f'{i}: {run_name}')
                    bin_thresh_segment = bin_thresh*0.75
                    main(
                        track_name, run_name, sr, cqt_window, s1, s2,
                        gap_interp, stab_hop_secs, min_stability_length_secs,
                        60, conv_filter_str, bin_thresh,
                        bin_thresh_segment, perc_tail,  gauss_sigma, cont_thresh,
                        etc_kernel_size, binop_dim, min_diff_trav, min_pattern_length_seconds,
                        min_in_group, match_tol, ext_mask_tol, n_dtw, thresh_dtw, thresh_cos, group_len_var, 
                        dupl_perc_overlap_inter, dupl_perc_overlap_intra, None,
                        None, partial_perc, top_n, write_plots,
                        write_audio, write_patterns, False, plot=False)


#########################
########## All ##########
#########################
failed = []
total = len(track_names)
i = 0
for t in track_names:
    # each date corresponds to performer
    # each performer has a different bin_thresh
    if '2018_11_13_am' in t:
            bin_thresh= 0.04
    elif '2018_11_13_pm' in t:
            bin_thresh = 0.06
    elif '2018_11_15' in t:
            bin_thresh= 0.04
    elif '2018_11_18_am' in t:
            bin_thresh=0.05
    elif '2018_11_19' in t:
            bin_thresh=0.05
    elif '2018_11_18_pm' in t:
            bin_thresh=0.04
    else:
        raise Exception('no params?')
    if t == '2018_11_13_am_Sec_3_P1_Anandabhairavi_V.2':
        bin_thresh = 0.02
    if t == '2018_11_18_am_Sec_6_P1_Atana_A':
        bin_thresh = 0.008
    if t == '2018_11_18_am_Sec_1_P1_Varaali':
        bin_thresh = 0.016
    if t == '2018_11_13_pm_Sec_5_P2_Atana':
        bin_thresh = 0.036
    if t == '2018_11_15_Sec_10_P1_Atana':
        bin_thresh = 0.016
    if t == '2018_11_15_Sec_2_P1_Anandabhairavi_B':
        bin_thresh = 0.008
    if t == '2018_11_15_Sec_8_P1_Bilahari':
        bin_thresh = 0.016

    i += 1
 #   try:
    title = f'{i}/{total} | Track name: {t}, bin_thresh: {bin_thresh}'
    print(title)
    print('-'*len(title))
    bin_thresh_segment = bin_thresh*0.75
    main(
        t, f'production_run', sr, cqt_window, s1, s2,
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
















t = '2018_11_18_pm_Sec4_P1_Bhairavi'
bin_thresh = 0.08
bin_thresh_segment = bin_thresh*0.75
main(
    t, f'test', sr, cqt_window, s1, s2,
    gap_interp, stab_hop_secs, min_stability_length_secs,
    60, conv_filter_str, bin_thresh,
    bin_thresh_segment, perc_tail,  gauss_sigma, cont_thresh,
    etc_kernel_size, binop_dim, min_diff_trav, min_pattern_length_seconds,
    min_in_group, match_tol, ext_mask_tol, n_dtw, thresh_dtw, thresh_cos, group_len_var, 
    dupl_perc_overlap_inter, dupl_perc_overlap_intra, None,
    None, partial_perc, top_n, write_plots,
    write_audio, write_patterns, False, plot=False)






#####################################
############ For Lara ###############
#####################################
from exploration.io import load_pkl
import pandas as pd
out_dir = '/Volumes/Shruti/FOR_LARA/icmpc/'
failed_tracks = []
for track_name in track_names:
    try:
        starts  = load_pkl(f'{out_dir}/{track_name}/results/icmpc/starts.pkl')
        lengths = load_pkl(f'{out_dir}/{track_name}/results/icmpc/lengths.pkl')

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

        df.to_csv(f'{out_dir}/{track_name}/results/icmpc/groups.csv', index=False)
    except:
        failed_tracks.append(track_name)


