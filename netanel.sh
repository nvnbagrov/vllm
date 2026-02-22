#!/bin/bash
HF_CHECKPOINT_DIR=/lustre/fsw/portfolios/llmservice/users/nhaber/repos/megatron-lm/omni/megatron_hf/
# HF_CHECKPOINT_DIR=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/nhaber/repos/megatron-lm/work/megatron_hf
python3 -m vllm.entrypoints.cli.main serve \
  --model $HF_CHECKPOINT_DIR \
  --trust-remote-code \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 16384 \
  --swap-space 8 \
  --data-parallel-size=8 \
  --port 30000 \
  --allowed-local-media-path / \
  --served-model-name model \
  --limit_mm_per_prompt '{"video": 0, "image": 1}' \
   --chat-template <(python3 -c "
import json
cfg = json.load(open('$HF_CHECKPOINT_DIR/tokenizer_config.json'))
tpl = cfg['chat_template']
tpl = tpl.replace(
   '{%- set enable_thinking = kw.enable_thinking if kw.enable_thinking is defined else True %}',
   '{%- set enable_thinking = False %}'
)
print(tpl)
")