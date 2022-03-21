# pad action space on single task s.t. can be fine-tuned later
CPUS=0-64
GAME=breakout
SEED=123
taskset -c $CPUS python main_mt.py --id Scratch-Pad18-Seed${SEED} --games $GAME --pad_action_space 18 \
    --T_max 10_000_000 --checkpoint_interval 100_000 --seed $SEED 


# simple scratch 
CPUS=0-64
GAME=pong 
for SEED in 123 321 213
do
taskset -c $CPUS python main_mt.py --game $GAME --id Scratch-Seed${SEED}
done