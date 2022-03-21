# single-task finetune
CPUS=0-64
GAME=ms_pacman
SEED=123
taskset -c $CPUS python main_mt.py --id FineTune-from-Alien-Seed${SEED} \
    --games $GAME --model results/Scratch-Alien/model.pth 

# freeze conv and reinit fc 
CPUS=64-128 
GAME=qbert
SEED=123
taskset -c $CPUS python main_mt.py --id FineTune-8task-v1 \
    --games $GAME --model /shared/mandi/rainbow_data/8Task-B256-NormRew/checkpoint_2000000.pth \
    --load_conv_only --reinit_fc 1 --unfreeze_conv_when 10_000