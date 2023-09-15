
# Collective Intelligence for 2D push Manipulations with Mobile Robots  
[So Kuroki](https://sites.google.com/view/sokuroki/home), Tatsuya Matsushima, Jumpei Arima, Hiroki Furuta, Yutaka Matsuo, Shixiang Shane Gu, and Yujin Tang

The University of Tokyo, Japan, Matsuo Institute, Japan, Google Research, Brain Team  
IEEE Robotics and Automation Letters (RA-L) with IROS  
[[Project page]](https://sites.google.com/view/ciom/home)
[[PDF]](https://ieeexplore.ieee.org/document/10080994)


## generating experts
### For gradient base difftaichi  
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve --algo action --env_name multi_bc_rope_6-v1  

## creating observation
python3 data_create.py

## train policy
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_bc_transformer_nearest.py --env_name multi_bc_rope_6-v1   
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_bc_transformer_nearest_random_delay.py --env_name multi_bc_rope_6-v1   
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_bc_mlp.py --env_name multi_bc_rope_6-v1  

## test
CUDA_VISIBLE_DEVICES=0,1 python3 -m plb.algorithms.solve_bc_transformer_nearest --algo action --env_name multi_bc_rope_6-v1   
CUDA_VISIBLE_DEVICES=0,1 python3 -m plb.algorithms.solve_bc_transformer_nearest_random_delay --env_name multi_bc_rope_6-v1   
CUDA_VISIBLE_DEVICES=0,1 python3 -m plb.algorithms.solve_bc_mlp --env_name multi_bc_rope_6-v1  

## test in real-world
rosrun roomba_control solve.py --env_name multi_bc_rope_{n}-v1 --real_demo  

# Acknowledgements
Our physics simulation is based on [PlasticineLab](https://github.com/hzaskywalker/PlasticineLab).  
Our ros system refer to [roomba hack](https://github.com/matsuolab/roomba_hack).
