SEED=123
CPUS=0-32
GAME=pong
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id OfflineRainbow --seed $SEED \
               --games $GAME \
               --load_dataset /home/mandi/rainbow_data/pong_rollouts  