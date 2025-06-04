DATASET=$1
ARCH=$2
ID=$3
GPU=$4
START=$5
END=$6
iter=best
python dynamics/test.py \
--gpus "${GPU}" \
--cfg "dynamics/outputs/phys/${DATASET}/${ID}/config.yaml" \
--predictor-arch "${ARCH}" \
--predictor-init "dynamics/outputs/phys/${DATASET}/${ID}/ckpt_${iter}.path.tar" \
--plot-image 0 \
--start_id "${START}" \
--end_id "${END}"

# BASE_COMMAND="sh dynamics/scripts/test_pred.sh PHYRE_1fps_p100n400 rpcin W0_rpcin_t5 0"

# for i in {0..24}; do
# 	$BASE_COMMAND $i $((i+1))
# done