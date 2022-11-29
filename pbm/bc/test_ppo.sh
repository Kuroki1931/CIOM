#!/bin/bash

ALGO='action'
ENV_NAME='multi_bc_rope_6-v'
POLICY_PARAMS_PATH='/root/roomba_hack/pbm/bc/network/dt_runs/bc_multi_bc_rope/seed_0/22-09-13-13-50-48/model_755000.pt'
PATH_NAME='../output'
PICTURE_NAME='pose'

POSE_NUM=50
EACH_POSE_NUM=10
STEPS=$(( ${POSE_NUM} / ${EACH_POSE_NUM} ))

export PYTHONPATH=../../pbm
set -euC

                                                                                     
for i in `seq 0 1 $(($STEPS-1))`
do
    # random sampling
    BASE=$(( ${i}*${EACH_POSE_NUM} ))
    seq 0 1 $((${EACH_POSE_NUM}-1)) | xargs -P ${EACH_POSE_NUM} -I{} bash each_test_ppo.sh {} \
                                                                                     ${ALGO} \
                                                                                     ${ENV_NAME} \
                                                                                     ${PATH_NAME} \
                                                                                     ${PICTURE_NAME} \
                                                                                     ${BASE} \
                                                                                     ${POLICY_PARAMS_PATH}
done

echo $SECONDS
