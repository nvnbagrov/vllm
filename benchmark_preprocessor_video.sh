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

#make sure FAST_PREPROCESS is set, and print the value
if [ -z "${FAST_PREPROCESS}" ]; then
  echo "FAST_PREPROCESS is not set"
  exit 1
fi
echo "FAST_PREPROCESS is set to ${FAST_PREPROCESS}"
fast_preprocessor=$FAST_PREPROCESS

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
# Video benchmarks use lower concurrency due to memory requirements
MEASURE_TTFT="--max-concurrency 1 --num-prompts 32"
########################################################################################################

frame_counts=(32 64 128)
resolutions=("512 512" "1920 1080" "3840 2160")
resolutions=("1920 1080")
# resolutions=("512 512")

for frame_count in "${frame_counts[@]}"; do
  for res in "${resolutions[@]}"; do
    read -r width height <<< "${res}"

    # skip 128 frames for 3840x2160 due to memory requirements
    if [ "${width}" -eq 3840 ] && [ "${height}" -eq 2160 ] && [ "${frame_count}" -eq 128 ]; then
      continue
    fi

    echo "Benchmarking FAST_PREPROCESS=${fast_preprocessor}, ${width}x${height}, ${frame_count} frames..."

    vllm bench serve \
    ${COMMON_BENCHMARK_PARAMS} \
    ${MEASURE_TTFT} \
    --random-input-len 128 \
    --random-output-len 128 \
    --random-mm-limit-mm-per-prompt '{"image": 0, "video": 1}' \
    --random-mm-bucket-config "{(${width}, ${height}, ${frame_count}): 1.0}" \
    > "${CHECKPOINT_SLUG}_${CHECKPOINT_PRECISION}_${width}x${height}_${frame_count}frames_video_c1_fast-${fast_preprocessor}.txt"

  done
done