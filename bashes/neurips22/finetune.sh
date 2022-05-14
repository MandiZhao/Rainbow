# scratch
CPUS=24-48
for SEED in 0 1 2 3 4  #123 312 231 321 213
do
for GAME in  pong assault battle_zone # ms_pacman beam_rider #
do
    taskset -c $CPUS python main_mt.py --target-update 2000 \
               --memory-capacity 100000 --T_max 100_000  \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --mlps 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id Scratch-DataEff \
               --games $GAME --learn_start 1600 \
               --checkpoint_interval 50_000                
done
done 

# 5 task
for TEST in pong assault battle_zone ms_pacman beam_rider #

for TEST in battle_zone ms_pacman beam_rider #ms_pacman beam_rider
do
MODEL=5-task-DataEff-B32-Seed123/checkpoint_1000000.pth
for SEED in {0..9}
do 
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100_000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id 5Task-FineTune --seed $SEED \
               --games $TEST --model $MODEL \
               --learn_start 1600 \
               --reset_sigmas --pad_action_space 18 --mlps 512
done 
done

# 10 task
for TEST in pong assault battle_zone ms_pacman beam_rider #

for TEST in ms_pacman beam_rider
do
MODEL=10-task-DataEff-B32-Seed123/checkpoint_1000000.pth
for SEED in {0..9}
do 
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100_000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id 10Task-FineTune --seed $SEED \
               --games $TEST --model $MODEL \
               --learn_start 1600 \
               --reset_sigmas --pad_action_space 18 --mlps 512
done 
done
