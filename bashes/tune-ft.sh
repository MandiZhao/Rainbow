# init sigma to large 
MODEL=results/8task-v2-DataEff-MT-B32-SepBuf-Seed123/checkpoint_500000.pth
GAME=jamesbond # up_n_down ms_pacman # pong battle_zone
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
               --reset_sigmas --noisy_std 0.01  
done

MODEL=results/8task-v2-DataEff-MT-B32-SepBuf-Seed123/checkpoint_500000.pth
GAME=jamesbond # up_n_down ms_pacman # pong battle_zone
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
               --id DataEff-NoNoise --model $MODEL --seed $SEED \
               --games $GAME \
               --learn_start 1600 \
               --noiseless --greedy_eps 0.2 --constant_greedy  # --load_conv_fc_h --unfreeze_conv_when 30_000 \
               #--act_greedy_until 100_000 
done


taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id DataEff-NoNoise --model $MODEL --seed $SEED \
               --games $GAME \
               --learn_start 1600 \
               --noiseless --load_conv_fc_h --unfreeze_conv_when 10_000 \
               --act_greedy_until 100_000 --greedy_eps 0.9

# scale rew by max 
CPUS=0-48
for SEED in 312 132 123
do
GAME=jamesbond
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id ScaleRewDataEff-Scratch --seed $SEED \
               --games $GAME \
               --learn_start 1600 --scale_rew '100k'   
done
#    --act_greedy_until 100_000 --greedy_eps 0.9 \