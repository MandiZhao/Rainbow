for SEED in 312 132 321 213 #123
do
for GAME in demon_attack #kangaroo battle_zone pong up_n_down ms_pacman jamesbond
do
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id DataEff --seed $SEED \
               --games $GAME --learn_start 1600 
done 
done 

# padded version for finetune
# 1task pretrain: amidar   assult       breakout bank_heist enduro      seaquest
# correspond to:  kangaroo demon_attack pong     ms_pacman  battle_zone up_n_down
for SEED in 123
do
for GAME in breakout bank_heist  #amidar assault breakout bank_heist enduro seaquest
do
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 500000 \
               --memory-capacity 500000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 10_000 --checkpoint_interval 50_000 \
               --id DataEff-Pad18 --seed $SEED \
               --games $GAME --learn_start 1600 --pad_action_space 18 
done 
done 

# MT

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
               --batch_size 64 --separate_buffer --num_games_per_batch 8 \
               --games 20-task --learn_start 32000 

# reptile
taskset -c $CPUS python main_mt.py --target-update 2000 \
               --learn_start 1600 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id DataEff-MT \
               --T_max 1_000_000 --checkpoint_interval 100_000 \
               --separate_buffer --games 8task-v2 \
               --reptile_k 5


# FT
GAME=kangaroo
GAME=demon_attack
MODEL=/shared/mandi/rainbow_data/8task-v2-DataEff-MT-Seed123/checkpoint_300000.pth
#amd2
MODEL=results/8task-v2-DataEff-MT-SepBuf-Seed123/checkpoint_400000.pth 

# pabamd1!

MODEL=results/8task-v2-DataEff-MT-B32-SepBuf-Seed123/checkpoint_500000.pth
MODEL=/shared/mandi/rainbow_data/20-task-DataEff-MT-B32-SepBuf-Seed123/checkpoint_1000000.pth

for GAME in battle_zone # up_n_down ms_pacman # pong battle_zone
do
for SEED in 312 132 123 321 213
do
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id DataEff-FineTune --seed $SEED \
               --games $GAME --model $MODEL \
               --learn_start 1600
done
done

MODEL=results/8task-v2-DataEff-MT-B32-SepBuf-Seed123/checkpoint_500000.pth
GAME=jamesbond # up_n_down ms_pacman # pong battle_zone
GSTEP=10_000
for SEED in 312 132 123 321 213
do
for EPS in 0.1 0.2 0.3 0.4
do 
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id DataEff-EpsGreedy --seed $SEED \
               --games $GAME --model $MODEL \
               --learn_start 1600 --act_greedy_until $GSTEP --greedy_eps $EPS

done

--load_conv_only --reinit_fc 0 --unfreeze_conv_when 20000 ;

# ft from 1 task
GAME1=Breakout
MODEL=results/${GAME1}-DataEff-Pad18-Seed123/checkpoint_500000.pth
GAME=pong
for SEED in 312 132 123 321 213
do
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id DataEff-FineTune-from-$GAME1 --seed $SEED \
               --games $GAME --model $MODEL \
               --learn_start 1600
done