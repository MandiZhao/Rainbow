
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 1_000_000 \
               --memory-capacity 100000 \
               --replay_frequency 8 \
               --multi-step 3 \
               --architecture data-efficient \
               --V_min 0 --V_max 10 --batch_size 512 \
               --noisy_std 0.5 \
               --learning_rate 0.00025 \
               --evaluation_interval 5000 \
               --id DataEff-512-Update8 --learn_start 1600 \
               --mlps 512 \
               --use_procgen --procgen_name coinrun --num_levels 100  
