
# Collective Intelligence for Object Manipulation with Mobile Robots

 - Run `python3 -m plb.algorithms.solve --algo [algo] --env_name [env_name] --path [output-dir]`. It will run algorithms `algo` for environment `env-name` and store results in `output-dir`. 

## one goal based trajectories obtimization
### For gradient base difftaichi  
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve --algo action --env_name two_roomba_deform_one_rope-v1 --picture_name pose
### For PPO 
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve --algo ppo --env_name two_roomba_deform_one_rope-v1 --picture_name pose
### For MPPI  
bash mppi.sh


## learning policy for random goal base

### ppo (random goal based) train
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve_ppo --env_name multi_bc_rope_6-v1500

### ppo test
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve_ppo_test --env_name multi_bc_rope_6-v1500

### behaviro cloning train
CUDA_VISIBLE_DEVICES=0,2 python3 scripts/train_bc.py --env_name multi_bc_rope_6-v1500  
CUDA_VISIBLE_DEVICES=0,2 python3 scripts/train_bc_each_act.py --env_name multi_bc_rope_6-v1500  
CUDA_VISIBLE_DEVICES=0,2 python3 scripts/train_bc_transformer.py --env_name multi_bc_rope_6-v1500  
CUDA_VISIBLE_DEVICES=0,2 python3 scripts/train_bc_transformer_mask.py --env_name multi_bc_rope_6-v1500  
CUDA_VISIBLE_DEVICES=0,2 python3 scripts/train_bc_transformer_mask_delay.py --env_name multi_bc_rope_6-v1500  
CUDA_VISIBLE_DEVICES=0,2 python3 scripts/train_bc_transformer_multi_task.py --env_name multi_bc_rope_6-v1500  
CUDA_VISIBLE_DEVICES=0,2 python3 scripts/train_bc_transformer_nearest.py --env_name multi_bc_rope_6-v1500  
CUDA_VISIBLE_DEVICES=0,2 python3 scripts/train_bc_transformer_nearest_random_delay.py --env_name multi_bc_rope_6-v1500  
CUDA_VISIBLE_DEVICES=0,2 python3 scripts/train_bc_mlp.py --env_name multi_bc_rope_6-v1500  

### behaviro cloning test
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve_bc_transformer_nearest --algo action --env_name multi_bc_rope_6-v1500  
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve_bc_transformer_nearest_random_delay --algo action --env_name multi_bc_rope_6-v1500

### real_demo
rosrun roomba_control solve.py --env_name multi_bc_rope_5_center-v1500 --real_demo  
rosrun roomba_control solve.py --env_name multi_bc_rope_4_center-v1500 --real_demo  
rosrun roomba_control solve.py --env_name multi_bc_rope_3-v1500 --real_demo  
rosrun roomba_control solve_real.py --env_name multi_bc_rope_5_center-v1500 --real_demo  
rosrun roomba_control solve_real2sim.py --env_name multi_bc_rope_5_center-v1500 --real_demo  


pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  
apt -y install nvidia-driver-430  
apt install nvidia-cuda-toolkit

https://qiita.com/yukoba/items/4733e8602fa4acabcc35  
https://zenn.dev/gomo/articles/7f6c28d002837c

