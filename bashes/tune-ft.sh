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
               --model $MODEL --seed $SEED \
               --games $GAME \
               --learn_start 1600 \
               --target-update 1000  --id DataEff-FrequentUpdate --no_wb
               #--noiseless --greedy_eps 0.2 --constant_greedy  # --load_conv_fc_h --unfreeze_conv_when 30_000 \
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
               --learn_start 1600 --no_wb #--scale_rew '100k'   
done
#    --act_greedy_until 100_000 --greedy_eps 0.9 \


# retrain a 2 layer fc model

CPUS=64-128

SEED=123
taskset -c $CPUS python main_mt.py --target-update 2000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --learning_rate 0.0001 \
               --evaluation_interval 50_000 \
               --id DataEff-MT-3MLP \
               --T_max 1_000_000 --checkpoint_interval 50_000 \
               --batch_size 64 --separate_buffer --num_games_per_batch 8 \
               --games 8task-v2 --learn_start 6400 \
               --evaluation_episodes 0 \
               --mlps 512 256 
