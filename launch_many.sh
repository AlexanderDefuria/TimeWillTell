#!/bin/bash

MODEL=(
#  "mistral3"
#  "deepseekcoder"
#  "codellama"
#  "llama3.1"
#  "llama3.2-1B"
#  "llama3.2-3B"
#  "gemma"
#  "qwen3"
  # GGUF zero-shot models
  "llama3.2-3B-gguf"
  "llama3.2-1B-gguf"
  "gemma3-4B-gguf"
  #"gemma3-27B-gguf"
  #"mistral3-7B-gguf"
  #"mistral3-24B-gguf"
  #"codellama-7B-gguf"
  #"deepseekcoder-7B-gguf"
  #"phi4-gguf"
  #"qwen3-coder-gguf"
  #"qwen3.5-27B-gguf"
  #"gpt-oss-20b-gguf"
  #"deepseekcoder-v2-16B-gguf"
)
ACCOUNT="def-cbelling-ab"
ACCOUNT="def-pbranco"
DATASET="megavul"
TOTAL_FOLDS=10
TAG="test-holdout"
FOLD_TYPE=(
"k_fold_cross_validation" 
"temporal_loo_block_cross_validation" 
"temporal_growing_window_cross_validation"
"random_loo_block_cross_validation" 
"random_growing_window_cross_validation" 
)
if [[ $(hostname) == *rorqual* ]]; then
  # ON RORQUAL
  SMALL_CPU="--cpus-per-task=4 --mem=31GB"
  NORMAL_CPU="--cpus-per-task=8 --mem=62GB"
  LARGE_CPU="--cpus-per-task=16 --mem=124GB"
else
  # ON FIR OR OTHER CLUSTER
  SMALL_CPU="--cpus-per-task=3 --mem=70GB"
  NORMAL_CPU="--cpus-per-task=6 --mem=140GB"
  LARGE_CPU="--cpus-per-task=12 --mem=280GB"
fi
SMALL_GPU="--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 $SMALL_CPU"
NORMAL_GPU="--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 $NORMAL_CPU"
LARGE_GPU="--gpus=h100:1 $LARGE_CPU"
TIME="--time=12:00:00"
ARRAY="--array=0-5%1"

is_small_model() {
  [[ "$1" == *4B-gguf* ]] && return 0;
  [[ "$1" == *1B-gguf* ]] && return 0;
  [[ "$1" == *3B-gguf* ]] && return 0;
  return 1
}

is_zero_shot_model() {
  [[ "$1" == *gguf* ]] && return 0 || return 1
}

is_large_model() {
  [[ "$1" == *large* ]] && return 0 || return 1
}

for model in "${MODEL[@]}"; do
  for fold_type in "${FOLD_TYPE[@]}"; do
    for fold in $(seq 0 $(( $TOTAL_FOLDS - 1))); do
      zero_shot_flag=""
      gpu_flag=""
      array_flag=""
      if is_zero_shot_model "$model"; then zero_shot_flag="--zero-shot --opro"; fi
      if is_zero_shot_model "$model"; then array_flag=""; else array_flag="$ARRAY"; fi
      if is_large_model "$model"; then gpu_flag="$LARGE_GPU"; else gpu_flag="$NORMAL_GPU"; fi
      if is_small_model "$model"; then gpu_flag="$SMALL_GPU"; fi
      echo "Launching $model , fold_type='$fold_type'"
      sbatch --job-name="$(($fold + 1))/$TOTAL_FOLDS-$model-$cw-$fold_type" \
        --account=$ACCOUNT \
        $TIME \
        $array_flag \
        $gpu_flag \
        -o "$model-$(($fold + 1))-of-$TOTAL_FOLDS-$cw-$fold_type-$TAG-%a.log" \
        TimeWillTell/gpu_drac.sh \
        $zero_shot_flag \
        --batch_size=2 \
        --grad_acc=32 \
        --model="$model" \
        --epochs=5 \
        --dataset="$DATASET" \
        --fold="$fold" \
        --total_folds=$TOTAL_FOLDS \
        --tag="$TAG" \
        --fold_type="$fold_type" 
    done
  done
done

echo "All processes have finished."


