# single-task finetune
CPUS=0-64
GAME=ms_pacman
SEED=123
taskset -c $CPUS python main_mt.py --id FineTune-from-Alien-Seed${SEED} \
    --games $GAME --model results/Scratch-Alien/model.pth 

GAME=pong
SEED=321
taskset -c $CPUS python main_mt.py --id FineTune-from-Breakout-Seed${SEED} \
    --games $GAME --model results/Breakout-Scratch-Pad18-Seed123/checkpoint_1500000.pth

# freeze conv and reinit fc 
CPUS=64-128 
GAME=pong
SEED=123
taskset -c $CPUS python main_mt.py --id FineTune-8task-v1 \
    --games $GAME --model /shared/mandi/rainbow_data/8Task-B256-NormRew/checkpoint_2000000.pth \
    --load_conv_only --unfreeze_conv_when 10_000 --reinit_fc 0

# load from 8task-v2
CPUS=64-128
GAME=pong
SEED=321
MODEL=8task-v2-MtTrain-Seed123-B256/checkpoint_2000000.pth
taskset -c $CPUS python main_mt.py --id FineTune-8task-v2 --games $GAME --model $MODEL 

# load for v1-late

SEED=213
GAME=qbert 
taskset -c $CPUS python main_mt.py --id FineTune-8task-v1-5M-Seed${SEED} \
    --games $GAME --model /shared/mandi/rainbow_data/8task-v1-B256/checkpoint_5000000.pth 
   