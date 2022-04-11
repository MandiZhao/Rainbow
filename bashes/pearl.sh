taskset -c $CPUS python main_mt.py --target-update 2000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 50_000 \
               --id DataEff-MT \
               --T_max 1_000_000 --checkpoint_interval 50_000 \
               --batch_size 32 --separate_buffer --num_games_per_batch 4 \
               --games 8-task --learn_start 128000 --pearl

# FT: set lr = 0
GAME=pong
for GAME in ms_pacman kangaroo jamesbond # pong battle_zone up_n_down 
do
MODEL=/shared/mandi/rainbow_data/8task-v2-PEARL-DataEff-MT-B32-SepBuf-Seed123/checkpoint_450000.pth
for SEED in 312 132 123 321 213
do
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0 \
               --evaluation_interval 5000 \
               --id DataEff-FineTune-PEARL --seed $SEED \
               --games $GAME --model $MODEL \
               --learn_start 1600 --pearl --evaluation_size 1000 --batch_size 2  
done
done
