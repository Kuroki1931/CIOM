#!/bin/bash

ALGO='action'
ENV_NAME='multi_bc_rope-v'
PATH_NAME='../output'
PICTURE_NAME='pose'

POSE_NUM=1
EACH_POSE_NUM=1
STEPS=$(( ${POSE_NUM} / ${EACH_POSE_NUM} ))

export PYTHONPATH=../../pbm
set -euC

                                                                                     
for i in `seq 0 1 $(($STEPS-1))`
do
    # random sampling
    BASE=$(( ${i}*${EACH_POSE_NUM} ))
    seq 0 1 $((${EACH_POSE_NUM}-1)) | xargs -P ${EACH_POSE_NUM} -I{} bash each_make_bc_datasets.sh {} \
                                                                                     ${ALGO} \
                                                                                     ${ENV_NAME} \
                                                                                     ${PATH_NAME} \
                                                                                     ${PICTURE_NAME} \
                                                                                     ${BASE}
done

echo $SECONDS
