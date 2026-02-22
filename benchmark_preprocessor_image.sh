export CHECKPOINT_PATH="${1:?Please provide checkpoint path as first argument}"
export CHECKPOINT_PRECISION="${2:?Please provide checkpoint precision (e.g., bf16) as second argument}"
export CHECKPOINT_SLUG="${3:?Please provide checkpoint slig as third argument}"
shift 3
EXTRA_VLLM_ARGS="$@"

echo "Passing extra VLLM args ${EXTRA_VLLM_ARGS}"

export VLLM_VIDEO_LOADER_BACKEND=opencv
export VLLM_LOG_FILE=logs/${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}.log

# https://github.com/vllm-project/vllm/issues/31579
export VLLM_FLOAT32_MATMUL_PRECISION=high
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

mkdir -p logs

# Start server
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model ${CHECKPOINT_PATH} \
  --trust-remote-code \
  --max-model-len 65536 \
  --mm-processor-cache-gb 0 \
  --dtype ${CHECKPOINT_PRECISION} \
  --no-enable-prefix-caching \
  ${EXTRA_VLLM_ARGS} > ${VLLM_LOG_FILE} 2>&1 &


sleep 5s
while true; do
    if grep -q "Application startup complete" "${VLLM_LOG_FILE}"; then
        echo "✅ Server startup complete!"
        break
    fi
    if grep -q "Exception" "${VLLM_LOG_FILE}"; then
        echo "❌ Found 'Exception' string in VLLM logs. Check the log at ${VLLM_LOG_FILE}."
        exit 1
    fi
    if grep -q "ERROR" "${VLLM_LOG_FILE}"; then
        echo "❌ Found 'ERROR' string in VLLM logs. Check the log at ${VLLM_LOG_FILE}."
        exit 1
    fi
    if grep -q "Error" "${VLLM_LOG_FILE}"; then
        echo "❌ Found 'Error' string in VLLM logs. Check the log at ${VLLM_LOG_FILE}."
        exit 1
    fi
    echo "Waiting for server startup..."
    sleep 10s
done

COMMON_BENCHMARK_PARAMS="--backend openai-chat --endpoint /v1/chat/completions --dataset-name random-mm --random-prefix-len 0 --ignore-eos --seed 42 --request-rate inf --model ${CHECKPOINT_PATH} --random-mm-base-items-per-request 1 --random-mm-num-mm-items-range-ratio 0"
########################################################################################################
MEASURE_THROUGHPUT="--max-concurrency 32 --num-prompts 256"
MEASURE_TTFT="--max-concurrency 1 --num-prompts 128"
########################################################################################################
# vllm bench serve ${COMMON_BENCHMARK_PARAMS} ${MEASURE_THROUGHPUT} --random-input-len 128  --random-output-len 128  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}'  --random-mm-bucket-config '{(1024, 1024, 1): 1.0}' > ${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}_1024_image_c32.txt
vllm bench serve ${COMMON_BENCHMARK_PARAMS} ${MEASURE_TTFT}       --random-input-len 128  --random-output-len 128  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}'  --random-mm-bucket-config '{(1024, 1024, 1): 1.0}' > ${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}_1024_image_c1.txt
########################################################################################################
# vllm bench serve ${COMMON_BENCHMARK_PARAMS} ${MEASURE_THROUGHPUT} --random-input-len 128  --random-output-len 128  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}'  --random-mm-bucket-config '{(2048, 2048, 1): 1.0}' > ${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}_2048_image_c32.txt
vllm bench serve ${COMMON_BENCHMARK_PARAMS} ${MEASURE_TTFT}       --random-input-len 128  --random-output-len 128  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}'  --random-mm-bucket-config '{(2048, 2048, 1): 1.0}' > ${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}_2048_image_c1.txt
########################################################################################################
# vllm bench serve ${COMMON_BENCHMARK_PARAMS} ${MEASURE_THROUGHPUT} --random-input-len 128  --random-output-len 128  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}'  --random-mm-bucket-config '{(4096, 4096, 1): 1.0}' > ${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}_4096_image_c32.txt
vllm bench serve ${COMMON_BENCHMARK_PARAMS} ${MEASURE_TTFT}       --random-input-len 128  --random-output-len 128  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}'  --random-mm-bucket-config '{(4096, 4096, 1): 1.0}' > ${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}_4096_image_c1.txt
########################################################################################################


## Results
echo "One 1024x1024 image ${CHECKPOINT_PRECISION}"
cat ${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}_1024_image_c1.txt  | grep "Median TTFT"
# cat ${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}_1024_image_c32.txt | grep "Output token throughput"

echo "One 2048x2048 image ${CHECKPOINT_PRECISION}"
cat ${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}_2048_image_c1.txt  | grep "Median TTFT"
# cat ${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}_2048_image_c32.txt | grep "Output token throughput"

echo "One 4096x4096 image ${CHECKPOINT_PRECISION}"
cat ${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}_4096_image_c1.txt  | grep "Median TTFT"
# cat ${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}_4096_image_c32.txt | grep "Output token throughput"
