# training 1 task agent for generalize
CPUS=192-255
CUDA_VISIBLE_DEVICES=5
SEED=123 
PAD=6
GAME=breakout # robotank bank_heist solaris
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --mlps 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id Scratch-DataEff-Pad${PAD} --seed $SEED \
               --games $GAME --learn_start 1600 --pad_action_space $PAD \
               --checkpoint_interval 10_000 

PAD=7
GAME=demon_attack
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --mlps 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id Scratch-DataEff-Pad${PAD} --seed $SEED \
               --games $GAME --learn_start 1600 --pad_action_space $PAD \
               --checkpoint_interval 10_000 


 
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 500_000 \
               --memory-capacity 500_000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --mlps 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 10000 \
               --id Scratch-DataEff-5Task --seed $SEED \
               --learn_start 9000 \
               --games 'breakout' 'demon_attack' 'robotank' 'bank_heist' 'solaris' \
               --separate_buffer 

# 1 to 1 finetune:
MODEL=results/8task-v2-DataEff-MT-B32-SepBuf-Seed123/checkpoint_500000.pth
GAME=jamesbond # up_n_down ms_pacman # pong battle_zone
for SEED in 312 132 123 # 321 213
do 
taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100000 \
               --memory-capacity 100000 \
               --replay_frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning_rate 0.0001 \
               --evaluation_interval 5000 \
               --id DataEff-ResetSig --seed $SEED \
               --games $GAME --model $MODEL \
               --learn_start 1600 \
               --reset_sigmas --noisy_std 0.01  
done


# # scratch 
# for SEED in 123 312 132 321 213 #123
# do
# for GAME in beam_rider assault # battle_zone pong ms_pacman  
# do
# taskset -c $CPUS python main_mt.py --target-update 2000 --T_max 100000 \
#                --memory-capacity 100000 \
#                --replay_frequency 1 \
#                --multi-step 20 \
#                --architecture data-efficient \
#                --mlps 256 \
#                --learning_rate 0.0001 \
#                --evaluation_interval 5000 \
#                --id Scratch-DataEff --seed $SEED \
#                --games $GAME --learn_start 1600 
# done 
# done 