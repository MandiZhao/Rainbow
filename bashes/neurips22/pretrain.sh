# train 1 task, pad 18!
GAME=breakout
TEST=pong 

GAME=demon_attack
TEST=assault


CPUS=129-192

GAME=robotank
TEST=battle_zone

GAME=bank_heist
TEST=ms_pacman

PAD=18
taskset -c $CPUS python main_mt.py --target-update 2000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --mlps 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id DataEff-Pad${PAD} \
               --games $GAME --learn_start 1600 --pad_action_space $PAD \
               --checkpoint_interval 50_000 \
               --T_max 500_000  


GAME=solaris
TEST=beam_rider
CUDA_VISIBLE_DEVICES=0

GAME=breakout 
TEST=pong
MODEL=results/${GAME}-DataEff-Pad18-Seed123/checkpoint_500000.pth
for SEED in 123 321 213 132 312 
do 
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100_000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id 1Task-FineTune --seed $SEED \
               --games $TEST --model $MODEL \
               --learn_start 1600 \
               --reset_sigmas --pad_action_space 18 --mlps 256 
done 


# 5 task train and 10 task train!

CPUS=128-255 
CPUS=0-128
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 1_000_000 \
               --memory-capacity 1_000_000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --mlps 512 \
               --learning_rate 0.0001 \
               --evaluation_interval 50_000 \
               --checkpoint_interval 50_000 \
               --id DataEff \
               --separate_buffer --num_games_per_batch 2  \
               --games 5-task --learn_start 9000  #  5-task
# reptile
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 1_000_000 \
               --memory-capacity 1_000_000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --mlps 512 \
               --learning_rate 0.0001 \
               --evaluation_interval 50_000 \
               --checkpoint_interval 50_000 \
               --id DataEff \
               --separate_buffer --num_games_per_batch 2  \
               --games 10-task --learn_start 16000 --reptile_k 5 

               --games 5-task --learn_start 9000 --reptile_k 5

               
# pearl 
CPUS=0-64
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 1_000_000 \
               --memory-capacity 1_000_000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --mlps 512 \
               --learning_rate 0.0001 \
               --evaluation_interval 50_000 \
               --checkpoint_interval 50_000 \
               --id DataEff \
               --separate_buffer --num_games_per_batch 2  \
               --games 5-task --learn_start 8000  --pearl --no_wb
               --games 10-task --learn_start 16000