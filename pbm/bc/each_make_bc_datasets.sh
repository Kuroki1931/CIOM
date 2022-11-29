#!/bin/bash

if [ $(( ${1} % 3)) -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ $(( ${1} % 3)) -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=1
elif [ $(( ${1} % 3)) -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=2
fi

VERSION=$(( ${1}+${6}+1 ))
ROPE_SEED=$(( ${VERSION} ))
NAME="${3}$VERSION"
echo ${ROPE_SEED}

set -euC

# # create goal state
# python3 -m plb.algorithms.solve \
#     --algo ${2} \
#     --env_name ${NAME} \
#     --path ${4} \
#     --picture_name ${5} \
#     --rope_seed ${VERSION} \
#     --create_grid_mass

# solve task
python3 -m plb.algorithms.solve \
    --algo ${2} \
    --env_name ${NAME} \
    --path ${4} \
    --rope_seed ${ROPE_SEED} \
    --picture_name ${5} \
    --seed ${VERSION}
