#!/bin/bash

ALGO='action'
ENV_NAME='multi_bc_rope_6-v'
PATH_NAME='../output'
PICTURE_NAME='pose'

POSE_NUM=50
EACH_POSE_NUM=5
STEPS=$(( ${POSE_NUM} / ${EACH_POSE_NUM} ))

export PYTHONPATH=../../pbm
set -euC

                                                                                     
for i in `seq 0 1 $(($STEPS-1))`
do
    # random sampling
    BASE=$(( ${i}*${EACH_POSE_NUM} ))
    seq 0 1 $((${EACH_POSE_NUM}-1)) | xargs -P ${EACH_POSE_NUM} -I{} bash each_test_bc_transformer_nearest.sh {} \
                                                                                     ${ALGO} \
                                                                                     ${ENV_NAME} \
                                                                                     ${PATH_NAME} \
                                                                                     ${PICTURE_NAME} \
                                                                                     ${BASE} 
done

echo $SECONDS
