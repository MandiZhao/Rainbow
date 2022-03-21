CPUS=64-128
GAME=8task-v2
SEED=123
taskset -c $CPUS python main_mt.py --id MtTrain-Seed${SEED} --games $GAME \
    --batch_size 256 --T_max 10_000_000  --evaluation_interval 100_000 --seed $SEED