import pathlib  # for finding files in directories
import sys  # for showing file loading progress bar
import librosa  # for audio processing
import librosa.display  # for plotting
import numpy as np  # for matrix and signal processing
import scipy  # for matrix processing
import cv2  # for image scaling
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  # for parallel processing
from scipy.spatial.distance import pdist, squareform # for eigenvector set distances
from statistics import median, mean  # for calculating median boundary estimation deviation

"""
Segmentation code derivative of:
     https://github.com/bmcfee/lsd_viz (MIT license)
with subsequent modifications from:
    https://github.com/chrispla/hierarchical_structure (MIT license)
and further modifications in this notebook.
"""

import skimage.io
import matplotlib.pyplot as plt


# @jit(forceobj=True)
def compute_laplacian(path, bins_per_octave, n_octaves, hop_length, downsampling):
    """Compute the Laplacian matrix from an audio file using
    its Constant-Q Transform.

    Args:
        path: filepath (str)
        bins_per_octave: number of bins per octave for CQT calculation
        n_octaves: number of octaves for CQT calculation

    Returns:
        L: Normalized graph Laplacian matrix (np.array)
        
    """

    # load audio
    y, sr = librosa.load(path, sr=16000, mono=True)

    # Compute Constant-Q Transform in dB
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y,
                                                   sr=sr,
                                                   bins_per_octave=bins_per_octave,
                                                   n_bins=n_octaves * bins_per_octave,
                                                   hop_length=hop_length)),
                                                   ref=np.max)

    # we won't beat-synchronize, to keep reference to user annoations without
    # depending on accuracy of beat tracking. We'll instead downsample the hopped
    # CQT and MFCCs by a factor of 15,625, to comform with the standard +-0.5
    # second tolerance for estimation
    Cdim = cv2.resize(C, (int(np.floor(C.shape[1]/downsampling)), C.shape[0]))

    # stack 4 consecutive frames
    Cstack = librosa.feature.stack_memory(Cdim, n_steps=4)

    # compute weighted recurrence matrix
    R = librosa.segment.recurrence_matrix(Cstack, width=3, mode='affinity', sym=True)

    # enhance diagonals with a median filter
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))
    Rf = librosa.segment.path_enhance(Rf, 15)

    # compute MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
    Mdim = cv2.resize(mfcc, (int(np.floor(mfcc.shape[1]/downsampling)), mfcc.shape[0]))

    # build the MFCC sequence matrix
    path_distance = np.sum(np.diff(Mdim, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    # get the balanced combination of the MFCC sequence matric and the CQT
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)
    A = mu * Rf + (1 - mu) * R_path

    # compute the normalized Laplacian
    L = scipy.sparse.csgraph.laplacian(A, normed=True)

    return L

# @jit(forceobj=True)
def decompose_laplacian(L, k1, k2):
    """Decompose Laplacian and make sets of its first integer k (k1 or k2) eigenvectors.
    For each set, compute its Euclidean self distance over time.

    Args:
        L: Laplacian matrix (np.array)
        k1: first-eigenvectors number for first set (int)
        k2: first-eigenvectors number for secomd set (int)

    Returns:
        distances: self distance matrix of each set of first eigenvectors (np.array, shape=(kmax-kmin, 512, 512))
    """

    # eigendecomposition
    evals, evecs = scipy.linalg.eigh(L)

    # eigenvector filtering
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

    # normalization
    Cnorm = np.cumsum(evecs**2, axis=1)**0.5

    # initialize set
    distances = []

    for k in [k1, k2]:
        # create set using first k normalized eigenvectors
        Xs = evecs[:, :k] / Cnorm[:, k-1:k]

        # get eigenvector set distances
        distance = squareform(pdist(Xs, metric='euclidean'))
        distances.append(distance)

    return np.asarray(distances)

# @jit(forceobj=True)
def get_representations(path):
    """Simple end-to-end call for getting the three representations
    
    Args:
        path: audio path (str)
        k1: first-eigenvectors number for first set (int)
        k2: first-eigenvectors number for secomd set (int)
    
    Returns:
        tuple:
            Laplacian: Normalized graph Laplacian matrix (np.array)
            approximation k1: self-distance matrix for k1 set (np.array)
            approximation k2: self-distance matrix for k2 set (np.array)
    """
    L = compute_laplacian(path=path, 
                          bins_per_octave=12*3, 
                          n_octaves=7,
                          hop_length=512, 
                          downsampling=15.625)

    d = decompose_laplacian(L=L, k1=4, k2=9)

    return (L, d[0], d[1])


"""
Checkerboard kernel and novelty function code with minor modifications from:
    FMP Notebooks, C4/C4S4_NoveltySegmentation.ipynb
    which is an implementation of:
        Jonathan Foote: Automatic audio segmentation using a measure of audio 
        novelty. Proceedings of the IEEE International Conference on Multimedia 
        and Expo (ICME), New York, NY, USA, 2000, pp. 452–455.
"""

# @jit(nopython=True)
def compute_kernel_checkerboard_gaussian(l=20, var=1, normalize=True):
    """Compute Guassian-like checkerboard kernel.

    Args:
        l: Parameter specifying the kernel size M=2*l+1 (int)
        var: Variance parameter determing the tapering (epsilon) (float)
        normalize: Normalize kernel (bool)

    Returns:
        kernel: Kernel matrix of size M x M (np.ndarray)
    """

    taper = np.sqrt(1/2) / (l * var)
    axis = np.arange(-l, l+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel

# @jit(nopython=True)
def compute_novelty(S, l=20, var=0.5, exclude=True):
    """Compute novelty function from SSM.

    Args:
        S: SSM (np.ndarray)
        l: Parameter specifying the kernel size M=2*l+1  (int)
        var: Variance parameter determing the tapering (epsilon) (float)
        exclude: Sets the first l and last l values of novelty function to zero (bool)

    Returns:
        nov (np.ndarray): Novelty function
    """

    kernel = compute_kernel_checkerboard_gaussian(l=l, var=var)
    N = S.shape[0]
    M = 2*l + 1
    nov = np.zeros(N)
    # np.pad does not work with numba/jit
    S_padded = np.pad(S, l, mode='constant')

    for n in range(N):
        # Does not work with numba/jit
        nov[n] = np.sum(S_padded[n:n+M, n:n+M] * kernel)
    if exclude:
        right = np.min([l, N])
        left = np.max([0, N-l])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov



# change to whatever audio you want to try here
audio_path = "/Volumes/Shruti/data/mir_datasets/saraga_carnatic/saraga1.5_carnatic/Akkarai Sisters at Arkay by Akkarai Sisters/Koti Janmani/Koti Janmani.multitrack-vocal.mp3"

# store all variables and features in this experiment in a dictionary for easier retrieval
# c1: Laplacian, c2: approximation 1, c3: approximation 2
exp = {'c1': {}, 'c2': {}, 'c3': {}}

# get representations and compute novelty
exp['c1']['rep'], exp['c2']['rep'], exp['c3']['rep'] = get_representations(audio_path)
for case in ['c1', 'c2', 'c3']:
    exp[case]['nov'] = np.abs(compute_novelty(exp[case]['rep'], l=18))

def save(matrix, filename):
	plt.imshow(matrix) #Needs to be in row,col order
	plt.savefig(filename)
	plt.close('all')

save(exp['c1']['rep'], 'c1.png')
save(exp['c2']['rep'], 'c2.png')
save(exp['c3']['rep'], 'c3.png')





# we want to define a threshold value for choosing novelty peaks to consider as boundaries
# for this experiment, let's define it as 70% of the difference between the global maximum
# and the non-zero global minimum of novelties.
ptp = 0.5 

for case in ['c1', 'c2', 'c3']:

    gmax = np.amax(exp[case]['nov'])
    gmin = np.amin(exp[case]['nov'][np.nonzero(exp[case]['nov'])])

    pt = gmin + (ptp * abs(gmax - gmin))
    exp[case]['pt'] = pt
    exp[case]['peaks'] = scipy.signal.find_peaks(exp[case]['nov'], height=pt)


fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for i, case in enumerate(['c1', 'c2', 'c3']):

    ax[0, i].matshow(exp[case]['rep'], cmap='Greys')
    ax[1, i].plot(exp[case]['nov'], color='black')
    ax[1, i].set_xlim(0, exp[case]['nov'].shape[0])
    ax[1, i].axhline(exp[case]['pt'], color='black', ls=':', alpha=0.5)
    for peak in exp[case]["peaks"][0]:
        ax[1, i].axvline(peak, color='r', ls='--', alpha=0.5)

fig.suptitle("Estimated structure of " + "audio 2", fontsize=18) 
plt.tight_layout()
plt.show()


exp['c1']['peaks'] = scipy.signal.find_peaks(exp['c1']['nov'], 
                                             height=exp['c1']['pt'], 
                                             distance=15)

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for i, case in enumerate(['c1', 'c2', 'c3']):

    ax[0, i].matshow(exp[case]['rep'], cmap='Greys')
    ax[1, i].plot(exp[case]['nov'], color='black')
    ax[1, i].set_xlim(0, exp[case]['nov'].shape[0])
    ax[1, i].axhline(exp[case]['pt'], color='black', ls=':', alpha=0.5)
    for peak in exp[case]["peaks"][0]:
        ax[1, i].axvline(peak, color='r', ls='--', alpha=0.5)

fig.suptitle("Estimated structure of " + "audio 2", fontsize=18) 
plt.tight_layout()
plt.show()                



# change audio directory here
audio_dir = "../salami-audio/"

unsorted_paths = []

for path in pathlib.Path(audio_dir).glob('**/*'):
    if path.is_file() and "wav" in str(path):
        unsorted_paths.append(path)

# let's get the song paths in order so that we can more easily
# compute subsets of the dataset

# get all numbers in parent folder names, and sort them
parent_numbers = []
for path in unsorted_paths:
    # get name of parent folder of song
    parent_name = int(str(pathlib.Path(*path.parts[-2:-1])))
    parent_numbers.append(parent_name)
parent_numbers = sorted(parent_numbers)

# resynthesize the file paths
paths = []
for number in parent_numbers:
    paths.append(pathlib.Path(audio_dir+str(number)+"/audio_mp3-to.wav"))






# dictionary with song parent folder name being the primary key, with secondary key 
# secondary key being the kernel size (9, 12, 15, 18, 21), and tertiary key "L", "a1", or "a2"
features = {}

# define song number, needs to be <=1446
song_no = 1446

if song_no > 1446:
    song_no = 1446

def process(path):
    
    # p_name = str(pathlib.Path(*path.parts[-2:-1]))
    d = {}

    L, a1, a2 = get_representations(str(path))

    for l in [9, 12, 15, 18, 21]:
        d[str(l)] = {}

        d[str(l)]["L"] = np.abs(compute_novelty(L, l=l))
        d[str(l)]["a1"] = np.abs(compute_novelty(a1, l=l))
        d[str(l)]["a2"] = np.abs(compute_novelty(a2, l=l))

    return d

results = Parallel(n_jobs=-2, verbose=1)(delayed(process)(path) for path in paths[:song_no])

# store them in the dictionary
for i, path in enumerate(paths[:song_no]):
    features[str(pathlib.Path(*path.parts[-2:-1]))] = results[i]



# change audio directory here
ann_dir = "../salami-ann/annotations/"

ann_paths = []

for path in pathlib.Path(ann_dir).glob('**/*'):
    if path.is_file() and "functions.txt" in str(path):
        ann_paths.append(path)

ann_fun = {}
for ann_path in ann_paths:
    p_name = str(pathlib.Path(*ann_path.parts[-3:-2]))

    # only create boundary dict if the other annotation hasn't created it first
    try:
        ann_fun[p_name]
    except KeyError:
        ann_fun[p_name] = {}

    ann_idx = str(ann_path)[-15]  # get annotation idx from textfile**1**_functions.txt
    boundaries = []
    with open(ann_path, "r") as f:
        lines = f.readlines()
        for l in lines:
            boundaries.append(float(l.split()[0]))

    ann_fun[p_name][ann_idx] = boundaries