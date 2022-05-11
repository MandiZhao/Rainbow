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

GAME=solaris
TEST=beam_rider
CUDA_VISIBLE_DEVICES=4
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

MODEL=results/${GAME}-DataEff-Pad${PAD}-B32-Seed123/checkpoint_500000.pth
for SEED in 123 321 213 132 312 
do 
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100_000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id 1Task-FineTune --seed $SEED \
               --games $TEST --model $MODEL \
               --learn_start 1600 \
               --reset_sigmas 
done 


# 5 task train and 10 task train!
