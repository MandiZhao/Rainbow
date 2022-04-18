# init sigma to large 
MODEL=results/8task-v2-DataEff-MT-B32-SepBuf-Seed123/checkpoint_500000.pth
GAME=kangaroo # up_n_down ms_pacman # pong battle_zone
for SEED in 312 132 123 # 321 213
do 
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id DataEff-ResetSig --seed $SEED \
               --games $GAME --model $MODEL \
               --learn_start 1600 \
               --reset_sigmas --noisy-std 0.01  
done