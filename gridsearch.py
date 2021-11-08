results_df = pd.DataFrame(columns=[
	'params',
	'n_groups', 
	'n_patterns',
	'max_n_in_group',
    'partial_match_precision',
    'partial_match_recall',
    'full_match_precision',
    'full_match_recall',
    'partial_group_precision',
    'full_group_precision',
	]
)

##################
### Parameters ###
##################
param_grid = {
	# Output paths of each step in pipeline
	'out_dir': [os.path.join("output", 'all_vocal')],

	'sim_filename': [os.path.join(out_dir, 'Koti Janmani_simsave.png')],
	'gauss_filename': [os.path.join(out_dir, 'Koti Janmani_gauss.png')],
	'edge_filename': [os.path.join(out_dir, 'Koti Janmani_edges.png')],
	'bin_filename': [os.path.join(out_dir, 'Koti Janmani_binary.png')],
	'cont_filename': [os.path.join(out_dir, 'Koti Janmani_cont.png')],
	'hough_filename': [os.path.join(out_dir, 'Koti Janmani_hough.png')],

	# Sample rate of audio
	'sr': [44100],

	# size in frames of cqt window from convolution model
	'cqt_window': [1984],

	# Take sample of data, set to None to use all data
	's1': [None], # lower bound index],
	's2': [None], # higher bound index],

	# pitch track extraction
	'frameSize': [2048], # For Melodia pitch extraction],
	'hopSize': [128], # For Melodia pitch extraction],
	'gap_interp': [250*0.001], # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]],
	'smooth': [7], # sigma for gaussian smoothing of pitch track [set to None to skip]]
	'audio_path': ['audio/Akkarai Sisters at Arkay by Akkarai Sisters/Koti Janmani/Koti Janmani.multitrack-vocal.mp3'],

	# Binarize raw sim array 0/1 below and above this value...
	'bin_thresh': [0.9,0.95,0.10,0.15],

	# Gaussian filter along diagonals with sigma...
	'gauss_sigma': [9,10,11,12,13,14,15,16,17,18],

	# After gaussian, re-binarize with this threshold
	'cont_thresh': [0.14,0.15,0.16],

	# Hough transform parameters
	'min_dist_sec': [0, 0.1, 0.2], # min dist in seconds between lines],
	'hough_threshold': [75, 80, 85],

	# Only search for lines between these angles (45 corresponds to main diagonal)
	'hough_low_angle': [44.5],
	'hough_high_angle': [45.5],

	# distance between consecutive diagonals to be joined
	'min_diff_trav': [25],	

	# Grouping diagonals
	'min_pattern_length_seconds': [2],
	'min_in_group': [3], # minimum number of patterns to be included in pattern group,

	# Minimum distance between start points (x,y) to be joined a segment group
	'same_seqs_thresh_secs': [1],

	# Exclusions
	'exclusion_functions': [[contains_silence, min_gap, too_stable]],

	# Evaluation
	'annotations_path': ['../carnatic-motifs/Akkarai_Sisters_-_Koti_Janmani_multitrack-vocal_-_ritigowla.TextGrid'],
	'eval_tol': [0.2]

}



import itertools
import tqdm

allNames = sorted(param_grid)
combinations = list(itertools.product(*(param_grid[Name] for Name in allNames)))
for param_set in tqdm.tqdm(combinations):
	param_dict = dict(zip(allNames, param_set))
	param_dict['min_length_cqt'] = param_dict['min_pattern_length_seconds']*param_dict['sr']/param_dict['cqt_window']
	param_dict['same_seqs_thresh'] =  int(param_dict['same_seqs_thresh_secs']*param_dict['sr']/param_dict['cqt_window'])


#	param_dict = {
#		'out_dir':out_dir,
#		'sim_filename':sim_filename,
#		'gauss_filename':gauss_filename,
#		'edge_filename':edge_filename,
#		'bin_filename':bin_filename,
#		'cont_filename':cont_filename,
#		'hough_filename':hough_filename,
#		'sr':sr,
#		'cqt_window':cqt_window,
#		's1':s1,
#		's2':s2,
#		'frameSize':frameSize,
#		'hopSize':hopSize,
#		'gap_interp':gap_interp,
#		'smooth':smooth,
#		'audio_path':audio_path,
#		'bin_thresh':bin_thresh,
#		'gauss_sigma':gauss_sigma,
#		'cont_thresh':cont_thresh,
#		'min_dist_sec':min_dist_sec,
#		'hough_threshold':hough_threshold,
#		'hough_low_angle':hough_low_angle,
#		'hough_high_angle':hough_high_angle,
#		'min_diff_trav':min_diff_trav,
#		'min_pattern_length_seconds':min_pattern_length_seconds,
#		'min_length_cqt':min_length_cqt,
#		'min_in_group':min_in_group,
#		'same_seqs_thresh_secs':same_seqs_thresh_secs,
#		'same_seqs_thresh':same_seqs_thresh,
#		'exclusion_functions':exclusion_functions,
#		'annotations_path':annotations_path,
#		'eval_tol':eval_tol
#	}

	###################################
	## Extract Pich Track From Audio ##
	###################################
	print('Extracting pitch track')
	pitch, raw_pitch, timestep = extract_pitch_track(audio_path, frameSize, hopSize, gap_interp, smooth, sr)

	##############
	## Binarize ##
	##############
	#print('Binarizing similarity matrix')
	X_bin = binarize(X, param_dict['bin_thresh'])

	#######################
	## Diagonal Gaussian ##
	#######################
	#print('Applying diagonal gaussian filter')
	X_gauss = diagonal_gaussian(X_bin, param_dict['gauss_sigma'])

	#############################
	## Contrast and Brightness ##
	#############################
	#print('Binarize gaussian blurred similarity matrix')
	X_cont = binarize(X_gauss, param_dict['cont_thresh'])

	###########
	## Hough ##
	###########
	#print('Applying Hough Transform')
	peaks = hough_transform(
		X_cont, param_dict['min_dist_sec'], param_dict['cqt_window'], 
		param_dict['hough_high_angle'], param_dict['hough_low_angle'], param_dict['hough_threshold'])

	#####################
	## Path Extraction ##
	#####################
	#print('Extracting paths')
	all_segments = get_all_segments(X_cont, peaks, param_dict['min_diff_trav'], param_dict['min_length_cqt'])

	####################
	## Group segments ##
	####################
	#print('Grouping Segments')
	all_groups = group_segments(all_segments, param_dict['same_seqs_thresh'])

	######################################
	## Convert to Pitch Track Timesteps ##
	######################################
	#print('Convert sequences to pitch track timesteps')
	starts_seq, lengths_seq = convert_seqs_to_timestep(all_groups, param_dict['cqt_window'], param_dict['sr'], timestep)

	###########################
	# Reduce/Apply Exclusions #
	###########################
	#print('Applying exclusion functions')
	starts_seq_exc, lengths_seq_exc = apply_exclusions(starts_seq, lengths_seq, param_dict['exclusion_functions'])

	starts_sec_exc = [[x*timestep for x in p] for p in starts_seq_exc]
	lengths_sec_exc = [[x*timestep for x in l] for l in lengths_seq_exc]

	################
	## Evaluation ##
	################
	#print('Evaluating')
	annotations_orig = load_annotations(annotations_path)
	annotations = evaluate_annotations(annotations_orig, starts_sec_exc, lengths_sec_exc, param_dict['eval_tol'])

	metrics = get_metrics(annotations, starts_sec_exc)

	metrics.update({'params': param_dict})
	results_df = results_df.append(metrics, ignore_index=True)












