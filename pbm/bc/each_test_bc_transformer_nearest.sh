#!/bin/bash

if [ $(( ${1} % 3)) -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ $(( ${1} % 3)) -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=1
elif [ $(( ${1} % 3)) -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=2
fi

VERSION=$(( ${1}+${6}+1500 ))
NAME="${3}$VERSION"
echo ${VERSION}

set -euC

# create goal state
# python3 -m plb.algorithms.solve \
#     --algo ${2} \
#     --env_name ${NAME} \
#     --path ${4} \
#     --picture_name ${5} \
#     --rope_seed ${VERSION} \
#     --create_grid_mass

# # test gradient base
# python3 -m plb.algorithms.solve \
#     --algo ${2} \
#     --env_name ${NAME} \
#     --path ${4} \
#     --rope_seed ${VERSION} \
#     --picture_name ${5}

# test using policy and gradient base
python3 -m plb.algorithms.solve_bc_transformer_nearest \
    --algo ${2} \
    --env_name ${NAME} \
    --path ${4} \
    --picture_name ${5} \

