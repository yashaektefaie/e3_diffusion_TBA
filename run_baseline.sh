#!/bin/bash
#SBATCH -c 1
#SBATCH -t 36:00:00
#SBATCH --mem=50G

#SBATCH -p kempner
#SBATCH --account kempner_mzitnik_lab
#SBATCH --gres=gpu:1

#SBATCH -o  logs/training_diffusion_5.out
#SBATCH -e logs/training_diffusion_5.err

#Change diffuson steps to 150 from 1000
#module load cuda/12.2.0-fasrc01
# conda init
# conda activate pgt
python main_qm9.py --model egnn_dynamics --lr 1e-4 --nf 192 --n_layers 6 --save_model True --diffusion_steps 150 --sin_embedding False --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 32 --normalize_factors [1,8,1] --conditioning tb_inhibition --dataset TB --exp_name TB_Diffusion

