# DifftaichiSim2Real

## Requirements  
### Build
./BUILD-DOCKER-IMAGE.sh  
### Run
./RUN-DOCKER-CONTAINER.sh


## Usage
 - Run `python3 -m plb.algorithms.solve --algo [algo] --env_name [env_name] --path [output-dir]`. It will run algorithms `algo` for environment `env-name` and store results in `output-dir`. 

### For gradient base difftaichi  
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve --algo action --env_name two_roomba_deform_one_rope-v1 --picture_name pose

### For PPO 
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve_async --algo ppo --env_name two_roomba_deform_one_rope-v1 --picture_name pose

### For MPPI  
bash mppi.sh
