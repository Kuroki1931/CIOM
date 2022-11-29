
# Collective Intelligence for Object Manipulation with Mobile Robots
[Project page](https://sites.google.com/view/collectiveintelligenceforobjec/home)
[arXiv](https://arxiv.org/pdf/2211.15136.pdf)


## generating experts
### For gradient base difftaichi  
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve --algo action --env_name multi_bc_rope-v1  
### For PPO 
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve --algo ppo --env_name multi_bc_rope-v1  

## creating observation
python3 data_create.py

## train policy
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_bc_transformer_nearest.py --env_name multi_bc_rope-v1   
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_bc_transformer_nearest_random_delay.py --env_name multi_bc_rope-v1   
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_bc_mlp.py --env_name multi_bc_rope-v1  

## test
CUDA_VISIBLE_DEVICES=0,1 python3 -m plb.algorithms.solve_bc_transformer_nearest --algo action --env_name multi_bc_rope-v1   
CUDA_VISIBLE_DEVICES=0,1 python3 -m plb.algorithms.solve_bc_transformer_nearest_random_delay --env_name multi_bc_rope-v1   
CUDA_VISIBLE_DEVICES=0,1 python3 -m plb.algorithms.solve_bc_mlp --env_name multi_bc_rope-v1  

## test in real-world
rosrun roomba_control solve.py --env_name multi_bc_rope_{n}-v1 --real_demo  
