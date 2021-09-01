## extract_motives

out_dir = os.path.join("output", 'akkarai')
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
            if any([(s1 > x - md and s1 < x + md) for x in pat_group]):
                continue
            if any([(s2 > x - md and s2 < x + md) for x in pat_group]):
                continue
            l_group.append(l)
            l_group.append(l)
            pat_group.append(s1)
            pat_group.append(s2)
        final_patterns.append(pat_group)
        lengths.append(l_group)

    return final_patterns, lengths

def seconds_convert(seq, sr):
    return [[x/sr for x in y] for y in seq]

starts, lengths = rearrange_patterns(patterns, 1984)
starts_secs, lengths_secs = seconds_convert(starts, 22050), seconds_convert(lengths, 22050)

# Sonify patterns if needed
if sonify:
    logging.info("Sonifying Patterns...")

    utils.sonify_patterns(wav_file, patterns, h)

utils.save_results_raw(patterns, outfile=outfile + "raw")

if is_ismir:
    ismir.plot_segments(X, segments)

# Alright, we're done :D
logging.info("Algorithm finished.")

