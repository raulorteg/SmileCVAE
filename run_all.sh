#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Smiles50RGB
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --
#BSUB -W 20:00
# request 32GB of system-memory
#BSUB -R "rusage[mem=32GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
mkdir logs
#BSUB -o logs/gpu_%J.out
#BSUB -e logs/gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/10.2
module load python3/3.7.7

# /appl/cuda/10.2/samples/NVIDIA_CUDA-10.2_Samples/bin/x86_64/linux/release/deviceQuery

pip3 install -r requirements.txt --user

n_epochs=2000
lr=0.0002
beta=0.5
batch_size=8
latent_size=20
resize=50

python3 cvae.py --n_epochs=$n_epochs --lr=$lr --beta=$beta  --batch_size=$batch_size --latent_size=$latent_size --resize=$resize
python3 training_plots.py --resize=$resize --latent=$latent_size
for degree in -3.0 -2.0 -1.5 -1.4 -1.3 -1.2 -1.1 -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 2.0 3.0
do
    python3 changing_smiles.py --degree=$degree --resize=$resize --latent=$latent_size
done

for degree in -3.0 -2.0 -1.5 -1.4 -1.3 -1.2 -1.1 -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 2.0 3.0
do
    python3 sample_prior.py --degree=$degree --resize=$resize --latent=$latent_size
done

for (( axis=0; axis <= $latent_size; ++axis ))
do
    python3 sample_across_axis.py --axis=$axis --resize=$resize --latent=$latent_size
done
