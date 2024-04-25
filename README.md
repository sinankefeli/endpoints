# Mistral Inference Usecase

Explore the various inference configurations using Mistral models. This document details the setup and execution processes for different model architectures and hardware setups.

## Available Configurations

1. **Mistral-7B on a single RTX 4090 GPU (Fast Inference)**
2. **Quantized Mistral-7B on CPU**
3. **Quantized Mistral-8x7B on a single RTX 4090 GPU and CPU** 

We recorded 1700 tokens / second for mistral-7B and 28.26 tokens / second for mixtral-8x7b-instruct-v0.1 on single 4090 + CPU.

## Environment Setup
We tested this code on the environment:
- [CUDA Driver](https://developer.nvidia.com/cuda-downloads)
- NVIDIA-SMI: 535.54.03
- Driver Version: 535.54.03
- CUDA Version: 12.2
- Toolkit: `cuda_12.3.r12.3/compiler.33567101_0`

## Installation

### Flash Attention

To enhance inference speed, install Flash Attention 2:
```bash
pip install ninja packaging wheel
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

For more details, view the [Flash Attention 2](https://tridao.me/publications/flash2/flash2.pdf).

## vLLM Arguments
Start the vllm server with the following command:
```
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --port=8002
```
For additional engine arguments, refer to [Engine Arguments](https://docs.vllm.ai/en/latest/models/engine_args.html).

## Sample Data and Analysis
Analyze job postings in `data` directory:
    Dataset: 1024 job postings.
    Input: Clean job descriptions.
    Average Tokens: 724.2 per description.

# Execution script
Run the inference script:
```
python -m inference
```

## llama.cpp Build Instructions
Build [llama.cpp](https://github.com/ggerganov/llama.cpp)
```
git clone https://github.com/ggerganov/llama.cpp.git
make
```

## Download Quantized Models
![GGUF](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-spec.png)
(Figure taken from [GGUF HuggingFace Page](https://huggingface.co/docs/hub/gguf))

Quantization Types: Choose from the types listed on the [Quantization Types](https://huggingface.co/docs/hub/gguf#quantization-types).

## Example Commands:
Download Mistral-7B-Instruct-v0.2-GGUF_Q6:
Mistral-7B-Instruct-v0.2-GGUF_Q6
```
huggingface-cli download \
    TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q6_K.gguf \
    --local-dir models \
    --local-dir-use-symlinks False
```

Mixtral-8x7B-Instruct-v0.1-GGUF_Q4_K_M
```
huggingface-cli download \
    TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf \
    --local-dir models \
    --local-dir-use-symlinks False
```

## Example Commands:


## Advanced Quantization
Resources:
- [GGUF HuggingFace Page](https://huggingface.co/docs/hub/gguf)
- [GGUF Github Repo](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
