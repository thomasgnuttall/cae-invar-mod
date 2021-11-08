# To run on SLURM
# sbatch get_started.sh --gres=gpu:1
# scontrol show job 2329876


# Install
pip install -e . 

# Train model
python train.py akkarai filelist_audio.txt config_cqt.ini

# Extract features
python convert.py akkarai filelist_audio.txt config_cqt.ini

# Creates self similarity matrix/output plots
python convert.py akkarai filelist_audio.txt config_cqt.ini --self-sim-matrix

# Repeated sections
python extract_motives.py akkarai -r 2 -th 0.01 -csv jku_csv_files.txt


cd /homedtic/tnuttall/asplab2/cae-invar

pip install -e .

python train.py akkarai filelist_audio.txt config_cqt.ini

# Extract features
python convert.py akkarai filelist_audio.txt config_cqt.ini

# Creates self similarity matrix/output plots
python convert.py akkarai filelist_audio.txt config_cqt.ini --self-sim-matrix

# Repeated sections
python extract_motives.py akkarai -r 2 -th 0.01 -csv jku_csv_files.txt