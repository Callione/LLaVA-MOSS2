# LLaVA-MOSS2: Baseline for “国际算法算例大赛-多模态大模型学科能力综合强化”

## Install

1. Clone this repository and navigate to LLaVA-MOSS2 folder

```bash
git clone https://github.com/Callione/LLaVA-MOSS2.git
cd LLaVA-MOSS2
```

2. Install Package

```Shell
conda create -n llava-moss2 python=3.10 -y
conda activate llava-moss2
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases

```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Train

LLaVA-MOSS2 adopts the training pipeline of LLaVA, which consists of two stages: (1) feature alignment stage: use 558K subset of the LAION-CC-SBU dataset to connect a *frozen pretrained* vision encoder to a *frozen LLM*; (2) visual instruction tuning stage: use [Llava-Instruct-665k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)

To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Construct new checkpoint:

Get a new MOSS2 checkpoint with the following two steps:

1. Download our base model MOSS-2 2.5B: [https://huggingface.co/fnlp/moss2-2_5b-chat](https://huggingface.co/fnlp/moss2-2_5b-chat)
2. Replace the origin `config.json` file in the checkpoint with the following content.

The second step changed the model_type to "llava_moss2", and added some multimodal hyperparameters.

```json
{
  "architectures": [
    "Moss2ForCausalLM"
  ],
  "attn_implementation": "eager",
  "bias": false,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 32768,
  "model_type": "llava_moss2",
  "num_attention_heads": 16,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pad_token_id": 2,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 2.0,
    "type": "dynamic"
  },
  "rope_theta": 1000000,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.39.0",
  "use_cache": true,
  "vocab_size": 137728,
  "freeze_mm_mlp_adapter": false,
  "freeze_mm_vision_resampler": false,
  "mm_hidden_size": 1024,
  "mm_projector_type": "mlp2x_gelu",
  "mm_resampler_type": null,
  "mm_use_im_patch_token": false,
  "mm_use_im_start_end": false,
  "mm_vision_select_feature": "patch",
  "mm_vision_select_layer": -2,
  "mm_vision_tower": "openai/clip-vit-large-patch14-336",
  "tune_mm_mlp_adapter": false,
  "tune_mm_vision_resampler": false,
  "unfreeze_mm_vision_tower": false,
  "use_mm_proj": true

}

```

### Pretrain (feature alignment)

Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions, data is [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main). Put the in directory `./playground/data/pretrain`, organize them as follows:

```
├── blip_laion_cc_sbu_558k.json
└── image
```

Training script with DeepSpeed ZeRO-2: [`./scripts/pretrain.sh`](./scripts/pretrain.sh).

- `--model_name_or_path`: set this with path to the checkpoint
- `--output_dir`: set this dir for saving the pretrained weights

Run pretraining with:

```shell
bash ./scripts/pretrain.sh
```

After pretraining, the parameter of the mlp projector will be saved in the output_dir, you should see a file named `mm_projector.bin`

### Visual Instruction Tuning

1. Prepare data

Please download the instruction tuning data of LLaVA-v1.5: [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) and place it in `./playground/data/`, and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [images](https://huggingface.co/datasets/weizhiwang/llava_v15_instruction_images/resolve/main/ocr_vqa_images_llava_v15.zip?download=true), after downloading, unzip it to the path `./ocr_vqa/images`
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `./playground/data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

2. Start training!

Training script with DeepSpeed ZeRO-2: [`./scripts/finetune.sh`](./scripts/finetune.sh).

set the following parameters:

* `--model_name_or_path`: path to constructed moss2 checkpoint
* `--pretrain_mm_mlp_adapter`: path to pretrained linear projector weight
* `--output_dir`: path for saving finetuned model weight

Run finetuning with:

```shell
bash ./scripts/finetune.sh
```

After finetuning, the model weight will be saved in the `output_dir` , you should see a file named `pytorch_model.bin`.

In order to get a model that can perform effective inference, you need to copy all the files in the base model [moss2-2_5b-chat](https://huggingface.co/fnlp/moss2-2_5b-chat) except `pytorch_model.bin` to the same level directory as the trained `pytorch_model.bin`. The organized directory structure is as follows:
```
.
├── config.json
├── configuration_moss2.py
├── generation_config.json
├── modeling_moss2.py
├── pytorch_model.bin
├── special_tokens_map.json
├── tokenization_moss2.py
├── tokenizer_config.json
├── tokenizer.model
└── trainer_state.json
```

## CLI Inference

Chat about images with LLaVA-MOSS2. It also supports multiple GPUs, 4-bit and 8-bit quantized inference. You can use your own trained model or the model we uploaded to [huggingface](https://huggingface.co/red-fox-yj/llava-moss2-2_5b-chat/tree/main), just replace the `--model-path`.

* `--conv-mode`　conversation mode should be set to align with finetuning stage

```Shell
python -m llava.serve.cli \
    --model-path red-fox-yj/llava-moss2-2_5b-chat \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit
```

<img src="images/demo_cli.gif" width="70%">

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon and the data we use.
