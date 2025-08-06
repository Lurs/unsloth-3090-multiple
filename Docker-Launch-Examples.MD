**Starts Docker with 3 GPUs and Training Script**

docker run --rm -it \
--gpus all \
-e NCCL_P2P_DISABLE=1 \
-e NCCL_DEBUG=INFO \
-e TORCH_DISTRIBUTED_DEBUG=DETAIL \
-e CUDA_LAUNCH_BLOCKING=1 \
-e TORCH_USE_CUDA_DSA=1 \
-e CUDA_VISIBLE_DEVICES=0,1,2 \
-v "$HOME/Desktop/Model Launchers/unsloth":/app/unsloth_run \
-v "/media/user/Models/Text/Qwen3-32B":/app/models/Qwen3-32B \
-v "/media/user/combined-alpaca-prompts/Final":/app/data/unsloth_datasets \
-v "$HOME/Desktop/Docker-Unsloth-Output":/app/output \
-w /app/unsloth_run \
rgilbreth/rtx-5xxx-unsloth-training:rtx-5xxx-unsloth-training \
/bin/bash -c "accelerate launch --multi_gpu --num_processes=3 unsloth_Accelerate-Docker.py --output_dir /app/output"


docker run --rm -it \
--gpus all \
-e NCCL_P2P_DISABLE=1 \
-e NCCL_DEBUG=INFO \
-e TORCH_DISTRIBUTED_DEBUG=DETAIL \
-e CUDA_VISIBLE_DEVICES=0,1,2,3 \
-e HF_TOKEN= \ 
-v /app/unsloth_run:/app/unsloth_run \
-v /app/output:/app/output \
-w /app/unsloth_run \
rgilbreth/rtx-5xxx-unsloth-training:rtx-5xxx-unsloth-training \
/bin/bash -c "accelerate launch --multi_gpu --num_processes=4 unsloth_Accelerate-Docker.py --output_dir /app/output"


**Starts Docker with 3 GPUs to Bash Command Line**

docker run --rm -it \
--gpus all \
-e NCCL_P2P_DISABLE=1 \
-e NCCL_DEBUG=INFO \
-e TORCH_DISTRIBUTED_DEBUG=DETAIL \
-e CUDA_LAUNCH_BLOCKING=1 \
-e TORCH_USE_CUDA_DSA=1 \
-e CUDA_VISIBLE_DEVICES=0,1,2 \
-v "$HOME/Desktop/Model Launchers/unsloth":/app/unsloth_run \
-v "/media/user/Models/Text/Qwen3-32B":/app/models/Qwen3-32B \
-v "/media/user/combined-alpaca-prompts/Final":/app/data/unsloth_datasets \
-v "$HOME/Desktop/Docker-Unsloth-Output":/app/output \
-w /app/unsloth_run \
rgilbreth/rtx-5xxx-unsloth-training:rtx-5xxx-unsloth-training \
/bin/bash 