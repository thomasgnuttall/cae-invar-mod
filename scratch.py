
%load_ext autoreload
%autoreload 2

## extract_motives
import os

out_dir = os.path.join("output", 'hpc')
input_filelist = os.path.join(out_dir, "ss_matrices_filelist.txt")
inputs = read_file(input_filelist)

args = {
        'tol': 0.01,
        'rho':2,
        'dom':'audio',
        'ssm_read_pk':False,
        'read_pk':False,
        'n_jobs':10,
        'tonnetz':False,
        'csv_files':'jku_csv_files_test.txt'
    }

if args['csv_files'] is not None:
    csv_files = read_file(args['csv_files'])
else:
    csv_files = None

## extract_motives.process_audio_poly
files=inputs
outdir=out_dir
tol=args['tol']
rho=args['rho']
domain=args['dom']
csv_files=csv_files
ssm_read_pk=args['ssm_read_pk']
read_pk=args['read_pk']
n_jobs=args['n_jobs']
tonnetz=args['tonnetz']

utils.ensure_dir(outdir)
if csv_files is None:
    csv_files = [None] * len(files)

## extract_motives.process_piece
#for wav in files:
wav = files[0]
csv = None



fn_ss_matrix=wav
outdir=outdir
tol=tol
domain=domain
ssm_read_pk=ssm_read_pk
read_pk=read_pk
tonnetz=tonnetz
rho=rho
csv_file=csv

f_base = os.path.basename(fn_ss_matrix)
base_name = os.path.join(outdir, f_base.split(".")[0] + ".seg")

#extractor.process(fn_ss_matrix, base_name, domain, csv_file=csv_file,
#                  tol=tol, ssm_read_pk=ssm_read_pk,
#                  read_pk=read_pk, tonnetz=tonnetz, rho=rho)
wav_file=fn_ss_matrix
outfile=base_name
domain=domain
csv_file=csv_file
tol=tol
bpm=None
ssm_read_pk=ssm_read_pk
read_pk=read_pk
tonnetz=tonnetz
rho=rho
is_ismir=False,
sonify=False

from complex_auto.motives_extractor import *
from complex_auto.motives_extractor.extractor import *

min_notes = 16
max_diff_notes = 5

# to process
if wav_file.endswith("wav"):
    # Get the correct bpm if needed
    if bpm is None:
        bpm = get_bpm(wav_file)
    
    h = bpm / 60. / 8.  # Hop size /8 works better than /4, but it takes longer
    # Obtain the Self Similarity Matrix
    X = compute_ssm(wav_file, h, ssm_read_pk, is_ismir, tonnetz)
elif wav_file.endswith("npy"):
    X = np.load(wav_file)
    X = prepro(X)
    if domain == "symbolic":
        h = .25 # 2. # for symbolic (16th notes)
    else:
        if bpm is None:
            bpm = get_bpm(wav_file)
        h = 0.0886 * bpm / 60

offset = 0

patterns = []
while patterns == []:
    # Find the segments inside the self similarity matrix
    logging.info("Finding segments in the self-similarity matrix...")
    max_diff = int(max_diff_notes / float(h))
    min_dur = int(np.ceil(min_notes / float(h)))
    #print min_dur, min_notes, h, max_diff
    if not read_pk:
        # [i, i+M, j, j+M]
        segments = []
        while segments == []:
            logging.info(("{0}: \ttrying tolerance %.2f" % tol).format(wav_file))
            segments = np.asarray(utils.find_segments(X, min_dur, th=tol, rho=rho))
            tol -= 0.001

        #utils.write_cPickle(wav_file + "-audio.pk", segments)
    else:
        segments = utils.read_cPickle(wav_file + "-audio.pk")

    # Obtain the patterns from the segments and split them if needed
    logging.info("Obtaining the patterns from the segments...")
    # [parent occurence, ]
    patterns = obtain_patterns(segments, max_diff)












def rearrange_patterns(patterns, h, max_diff=5):
    lengths = []
    final_patterns = []
    md = max_diff*h
    for p_group in patterns:
        l_group = []
        pat_group = []
        for p in p_group[1:]:
            l = (p[1]-p[0])*h
            s1 = p[0]*h
            s2 = p[3]*h
            #if any([(s1 > x - md and s1 < x + md) for x in pat_group]):
             #   continue
            #if any([(s2 > x - md and s2 < x + md) for x in pat_group]):
             #   continue
            l_group.append(l)
            l_group.append(l)
            pat_group.append(s1)
            pat_group.append(s2)
        final_patterns.append(pat_group)
        lengths.append(l_group)

    return final_patterns, lengths

def seconds_convert(seq, sr, return_type=float):
    return [[return_type(x/sr) for x in y] for y in seq]

sr = 44100
cqt_window = 1984
starts, lengths = rearrange_patterns(patterns, cqt_window)
starts_secs, lengths_secs = seconds_convert(starts, sr), seconds_convert(lengths, sr)


import librosa

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

import essentia.standard as estd

import pandas as pd 
from scipy.ndimage import gaussian_filter1d

# Save output
def interpolate_below_length(arr, val, gap):
    """
    Interpolate gaps of value, <val> of 
    length equal to or shorter than <gap> in <arr>
    
    :param arr: Array to interpolate
    :type arr: np.array
    :param val: Value expected in gaps to interpolate
    :type val: number
    :param gap: Maximum gap length to interpolate, gaps of <val> longer than <g> will not be interpolated
    :type gap: number

    :return: interpolated array
    :rtype: np.array
    """
    s = np.copy(arr)
    is_zero = s == val
    cumsum = np.cumsum(is_zero).astype('float')
    diff = np.zeros_like(s)
    diff[~is_zero] = np.diff(cumsum[~is_zero], prepend=0)
    for i,d in enumerate(diff):
        if d <= gap:
            s[int(i-d):i] = np.nan
    interp = pd.Series(s).interpolate(method='linear', axis=0)\
                         .ffill()\
                         .bfill()\
                         .values
    return interp

frameSize = 2048 # For Melodia pitch extraction
hopSize = 128 # For Melodia pitch extraction
gap_interp = 250*0.001 # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]
smooth = 7 # sigma for gaussian smoothing of pitch track [set to None to skip]


audio_path = 'audio/Akkarai Sisters at Arkay by Akkarai Sisters/Koti Janmani/Koti Janmani.multitrack-vocal.mp3'
audio_loaded, _ = librosa.load(audio_path, sr=sr)

# Run spleeter on track to remove the background
separator = Separator('spleeter:2stems')
audio_loader = AudioAdapter.default()
waveform, _ = audio_loader.load(audio_path, sample_rate=sr)
prediction = separator.separate(waveform=waveform)
clean_vocal = prediction['vocals']

# Prepare audio for pitch extraction
audio_mono = clean_vocal.sum(axis=1) / 2
audio_mono_eqloud = estd.EqualLoudness(sampleRate=sr)(audio_mono)

# Extract pitch using Melodia algorithm from Essentia
pitch_extractor = estd.PredominantPitchMelodia(frameSize=frameSize, hopSize=hopSize)
raw_pitch, _ = pitch_extractor(audio_mono_eqloud)
raw_pitch_ = np.append(raw_pitch, 0.0)
time = np.linspace(0.0, len(audio_mono_eqloud) / sr, len(raw_pitch))

timestep = time[4]-time[3] # resolution of time track

# Gap interpolation
if gap_interp:
    print(f'Interpolating gaps of {gap_interp} or less')
    raw_pitch = interpolate_below_length(raw_pitch_, 0, int(gap_interp/timestep))
    
# Gaussian smoothing
if smooth:
    print(f'Gaussian smoothing with sigma={smooth}')
    pitch = gaussian_filter1d(raw_pitch, smooth)
else:
    pitch = raw_pitch[:]

silent_mask = pitch == 0

# st_perm = itertools.combinations(enumerate(starts_secs_round), 2)

# groupings = {i:[] for i in range(len(starts_secs_round))}

# for ((i1, stl1), (i2, stl2)) in st_perm:
#     if any([x in stl2 for x in stl1]):
#         groupings[i1] += [i2]
#         groupings[i2] += [i1]


starts_seq, lengths_seq = seconds_convert(starts_secs, timestep, int), seconds_convert(lengths_secs, timestep, int)


####################
# Apply Exclusions #
####################
min_length = 3
min_num = 3
max_zero_prop = 0.2
starts_seq_cut = []
lengths_seq_cut = []
for i, seq in enumerate(starts_seq):
    this_seqs = []
    this_len  = []
    for j, s in enumerate(seq):
        length = lengths_seq[i][j]
        pseq = pitch[s:s+length]
        prop_zero = float(sum(pseq==0))/len(pseq)
        if prop_zero <= max_zero_prop:
            this_seqs.append(s)
            this_len.append(length)

    seqs_rounded = [(i,round(x,-1)) for i,x in enumerate(this_seqs)]
    seqs_sorted = [x for x in sorted(seqs_rounded, lambda y: -lengths_seq_cut[y[0]])]
    lengths_sorted = sorted(seqs_rounded, lambda y: -y)





    if len(this_seqs) > min_num and any([x*timestep > min_length for x in this_len]):
        starts_seq_cut.append(this_seqs)
        lengths_seq_cut.append(this_len)
print(f'{len(starts_seq_cut)} groups')




############
# Database #
############
from exploration.utils import sql
from credentials import settings
import psycopg2

def insertResults(records, params):
    try:
        connection = psycopg2.connect(**settings)

        cursor = connection.cursor()

        # Update single record now
        sql_insert_query = """ 
        INSERT INTO results 
        (patternnumber, recordingid, elementnumber, durationelements, starttimeseconds, durationseconds, patterngroup, rankingroup)
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.executemany(sql_insert_query, records)
        connection.commit()
        count = cursor.rowcount
        print(count, "Record Updated successfully ")

    except (Exception, psycopg2.Error) as error:
        print("Error in update operation", error)

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

def insertSimilarity(records, params):
    try:
        connection = psycopg2.connect(**settings)

        cursor = connection.cursor()

        # Update single record now
        sql_insert_query = """ 
        INSERT INTO similarity 
        (patternnumberone, patternnumbertwo, similarityname, similarity)
        VALUES(%s, %s, %s, %s)"""
        cursor.executemany(sql_insert_query, records)
        connection.commit()
        count = cursor.rowcount
        print(count, "Record Updated successfully ")

    except (Exception, psycopg2.Error) as error:
        print("Error in update operation", error)

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

recording_id = 'brovabarama'
records = []

pattern_num = 0
pattern_num_lookup = {}
for i, seq in enumerate(starts_seq_cut):
    for j, s in enumerate(seq):
        length = lengths_seq_cut[i][j]
        length_secs = round(length*timestep,2)
        start_time_secs = round(s*timestep,2)
        records.append((pattern_num, recording_id, s, length, start_time_secs, length_secs, i, j))
        pattern_num_lookup[pattern_num] = (i,j)
        pattern_num += 1


insertTable(records, settings)



import itertools
similarities = []
for s1, s2 in itertools.combinations(pattern_num_lookup.keys(), 2):
    for n in ['cosine', 'dtw', 'eucliedean']:
        similarities.append((s1, s2, n, np.random.random()))





# train model more
    # - parameters
        # Tune frequency bands 
            # for this music, perhaps a standard fourier transform would work better?
            # what is fmin
            # how many octaves
            # frequency distribution across all tracks can inform parameters
    # - check graphs
    # - no further test performance increase after ~1250 epochs

# link features to annotations from Lara for phrase onset detection
    # load features and annotations

from complex_auto.util import load_pyc_bz
import textgrid

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


def transform_features(features):
    amp_arr = features[0].detach().numpy()
    phase_arr = features[1].detach().numpy()
    
    nbins = amp_arr.shape[1]
    
    amp_cols = [f'amp_{i}' for i in range(nbins)]
    phase_cols = [f'phase_{i}' for i in range(nbins)]
    
    amp_df = pd.DataFrame(amp_arr, columns=amp_cols)
    phase_df = pd.DataFrame(phase_arr, columns=phase_cols)

    df = pd.concat([amp_df, phase_df], axis=1)
    df['window_num'] = df.index
    return df


def second_to_window(onset, sr, hop_size):
    onset_el = onset*sr
    window_num = math.floor(onset_el/hop_size)
    return window_num


features_paths = [
    'output/hpc/Koti Janmani.multitrack-vocal.mp3_repres.pyc.bz',
    'output/hpc/Shankari Shankuru.multitrack-vocal.mp3_repres.pyc.bz',
    'output/hpc/Sharanu Janakana.multitrack-vocal.mp3_repres.pyc.bz'
]
annotations_paths = [
    '../carnatic-motifs/Akkarai_Sisters_-_Koti_Janmani_multitrack-vocal_-_ritigowla.TextGrid',
    '../carnatic-motifs/Akkarai_Sisters_-_Shankari_Shankuru_multitrack-vocal_-_saveri.TextGrid',
    '../carnatic-motifs/Salem_Gayatri_Venkatesan_-_Sharanu_Janakana_multitrack-vocal_-_bilahari_copy.TextGrid'
]


all_features = pd.DataFrame()

for i,(fp, ap) in enumerate(zip(features_paths, annotations_paths)):

    # array of [amplitude, phase]
    features_raw = load_pyc_bz(fp)
    features = transform_features(features_raw)
    annotations = load_annotations(ap)

    hop_size = cqt_window # 1984

    annotations['window_num'] = annotations['s1'].apply(lambda y: second_to_window(y, sr, hop_size))

    features['is_onset'] = features['window_num'].isin(annotations['window_num'])
    features['is_test'] = i==2
    all_features = all_features.append(features, ignore_index=True)




# Classification
import lightgbm as lgb
from scipy.stats import randint as sp_randint
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     RandomizedSearchCV, TimeSeriesSplit,
                                     cross_val_score, train_test_split)
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

def random_float_inrange(N,a=0.005,b=0.1):
    return[((b - a) * np.random.random_sample()) + a for _ in range(N)]


#df_train, df_test = train_test_split(all_features, test_size=0.4, random_state=42)
df_train = all_features[all_features['is_test']==False]
df_test = all_features[all_features['is_test']==True]

# resample
# Resample to account for huge sparsity
pos_frame = df_train[df_train['is_onset']==1]
neg_frame = df_train[df_train['is_onset']!=1]

while sum(df_train['is_onset'])/len(df_train) < 0.3:
    print(sum(df_train['is_onset'])/len(df_train))
    random_rec = pos_frame.sample(1000)
    df_train = df_train.append(random_rec, ignore_index=True)

# shuffle frame
df_train = df_train.iloc[np.random.permutation(len(df_train))].reset_index(drop=True)


feat_names = [c for c in df_train if c not in ['is_onset', 'window_num', 'is_test']]

X_train = df_train[feat_names].values
y_train = df_train['is_onset'].values

X_test = df_test[feat_names].values
y_test = df_test['is_onset'].values


param_dist = {'reg_sqrt':[True],
             'learning_rate':[0.001,0.01,0.1, 0.5],
             'max_depth':[2,4,8,12],
             'min_data_in_leaf':[1,5,10],
             'num_leaves':[5,10,15,20,25],
             'n_estimators':[100,200,300,400],
             'colsample_bytree':[0.6, 0.75, 0.9]}


# Final features from gridsearch
final_params = {
    'colsample_bytree': 0.6463615939999198,
    'learning_rate': 0.1280212488889668,
    'max_depth': 40,
    'min_data_in_leaf': 27,
    'n_estimators': 982,
    'num_leaves': 46,
    'reg_sqrt': True
}

lgb_model = lgb.LGBMClassifier(**final_params)

# Gridsearch
lgb_model = lgb.LGBMClassifier()
lgb_model = RandomizedSearchCV(lgb_model, param_distributions=param_dist,
                               n_iter=1000, cv=3, n_jobs=-1,
                               scoring='recall', random_state=42)

lgb_model.fit(X_train, y_train)




y_pred = lgb_model.predict(X_test)
for scorer in recall_score, precision_score, f1_score, roc_auc_score:
    print(f'{scorer.__name__}: {scorer(y_test, y_pred)}')


importances = list(sorted(zip(feat_names, lgb_model.feature_importances_), key=lambda y: -y[1]))
importances[:10]





# black out similarity grid based on
    # consonant onset
    # silence
    # stability

# link db to ladylane

















sql("""

    SELECT 
    results.patternnumber,
    results.patterngroup,
    results.rankingroup,
    results.starttimeseconds,
    results.durationseconds

    FROM results

    WHERE results.recordingid = 'brovabarama'
    AND results.patterngroup = 1


""")



sql("""

    SELECT 

    patternnumberone,
    patternnumbertwo,
    similarity,
    similarityname

    FROM similarity
    WHERE similarityname = 'cosine'
    AND (patternnumberone = 4 OR patternnumbertwo = 4)

    ORDER BY similarity
    
""")



















insertSimilarity(similarities, settings)

#######################
# Output subsequences #
#######################
from exploration.visualisation import plot_all_sequences, plot_pitch
from exploration.io import write_all_sequence_audio

plot_kwargs = {
    'yticks_dict':{},
    'cents':True,
    'tonic':195.997718,
    'emphasize':{},#['S', 'S^'],
    'figsize':(15,4)
}

out_dir = 'output/hpc/test/'

plot_all_sequences(pitch, time, lengths_seq_cut, starts_seq_cut, out_dir, clear_dir=True, plot_kwargs=plot_kwargs)
write_all_sequence_audio(audio_path, starts_seq_cut, lengths_seq_cut, timestep, out_dir)



# x Exclusion mask apply
# - Output patterns and audio with plots
# - Store in database
    # - recording_id, seq_num, duration_seq, seq_sec, duration_sec, group number, group rank
# - Quick get next pattern





