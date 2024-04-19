# Exploiting Diffusion Prior for Generalizable Dense Prediction

[Hsin-Ying Lee](https://shinying.github.io), [Hung-Yu Tseng](https://hytseng0509.github.io), [Hsin-Ying Lee](http://hsinyinglee.com), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang)

**CVPR 2024**

[Project Page](https://shinying.github.io/dmp) | [arXiv](https://arxiv.org/abs/2311.18832)

This is the official implementation of *Exploiting Diffusion Prior for Generalizable Dense Prediction*.

## Quick Start

### Installation

Our implementation is based on Python 3.10 and CUDA 11.3.

**Required**

```
diffusers==0.20.0
pytorch==1.12.1
torchvision==0.13.1
transformers==4.31.0
```

**Optional**

```shell
accelerate  # for training
gradio      # for demo
omegaconf   # for configuration
xformers    # for acceleration
```

### Checkpoints

We provide the model weights of five tasks for reproducing the results in the paper. These checkpoints are trained with 10K synthesized bedroom images, prompts, and pseudo ground truths.

Besides, for normal and depth prediction, we provide the weights trained with more diverse scenes and without prompts, which are more suitable for practical use cases.

Download the weights from [this google drive](https://drive.google.com/drive/folders/1R7H2x3ETkxs5bRx9gzKo4ngxtFJfGPAv?usp=sharing) and place them in the root directory.

### Run

For checkpoints with `-notext`, set `disable_prompts=True`.

```python
from PIL import Image
from pipeline import Pipeline

LORA_DIR = 'ckpt/normal-scene100-notext'
disable_prompts = LORA_DIR.endswith('-notext')
ppl = Pipeline(
    disable_prompts=disable_prompts,
    lora_ckpt=LORA_DIR,
    device='cuda',
    mixed_precision='fp16',
)
img = Image.open('/path/to/img')
```

For depth prediction,

```python
output_np_array = ppl(img, inference_step=5, target_mode='F')
```

Otherwise,

```python
output_pil_img = ppl(img, inference_step=5, target_mode='RGB')
```

Alternatively, we provide [Gradio](https://www.gradio.app) demo. You can launch it with

```shell
python app.py
```

and access the app at `localhost:7860`.

---

## Data Generation

### Images

We conduct the experiments with synthetic images, so we can control and analyze the performance of different data domains. We first generate prompts with scene keywords. Then we generate images with the prompts.

To generate **prompts**,

```shell
python tools/gencap.py KEYWORD -n NUMBER_OF_PROMPTS -o OUTPUT_TXT
```

`KEYWORD` can be a single word or a text file containing multiple words separated by lines.

To generate **images**,

```shell
python tools/txt2img.py --from-file PROMPTS_TXT --output OUTPUT_DIR --batch-size BSZ
```

These two scripts are some wrappers of huggingface's [transformers](https://github.com/huggingface/transformers) and [diffusers](https://github.com/huggingface/diffusers).

Then make a meta file to record images and prompts. Prompts are not necessary if you set `disable-prompts` (see the section of training).

```shell
python tools/makemeta.py --imgs IMAGE_DIR [--captions PROMPTS]
```

It collects the png and jpg files in `IMAGE_DIR`, sort them by their file names, and generates a `metadata.jsonl` in `IMAGE_DIR` with the same format as huggingface's [ImageFolder](https://huggingface.co/docs/datasets/image_dataset#image-captioning). If prompts are provided, it should be in the same order as the file names.

### Pseudo Ground Truths

Then we generate pseudo ground truths with the following code bases.

* surface normals: [3DCommonCorruptions](https://github.com/EPFL-VILAB/3DCommonCorruptions)

* depths: [ZoeDepth](https://github.com/isl-org/ZoeDepth)

* albedo and shading: [PIE-Net](https://github.com/Morpheus3000/PIE-Net)

* semantic segmentation: [EVA-02](https://github.com/baaivision/EVA)

For **normals**, **albedo** and **shading**, clone the repo, set up the environments, and put `getnorm.py` and `getintr.py` in each directory.

For **depths**, `getdepth.py` can be run in isolation.

```shell
python tools/get{norm,depth,intr}.py -i INPUT_IMG_DIR -o OUTPUT_DIR
```

These scripts store the predictions in [lmdb](https://lmdb.readthedocs.io) by default. The keys of predictions are the file names without extensions. The keys of albedo and shading outputs get an extra `-r` (**r**eflectance) and`-s` (**s**hading) suffix. Use `--save-files` to save outputs in files.

For **semantic segmentation**, generate segmentation maps with `eva02_L_ade_seg_upernet_sz512` in [EVA-02](https://github.com/baaivision/EVA/tree/master/EVA-02/seg).

<details>
<summary>Detailed instructions</summary>

1. Download `eva02_L_ade_seg_upernet_sz512.pth`

2. Make a directory with an arbitrary name, e.g. `dummy`, and make another directory named `images` under it.
   
   ```shell
   mkdir -p dummy/images
   ```

3. Link the directory of input images as `validation` in `dummy/images`
   
   ```shell
   ln -s INPUT_IMG_DIR dummy/images/validation
   ```

4. Modify `data_root` at line 3 in `configs/_base_/datasets/ade20k.py` to be `dummy`

5. Run `test.py` in the [EVA-02 repo](https://github.com/baaivision/EVA/tree/master/EVA-02/seg) with
   
   ```shell
   python test.py \
      configs/eva02/upernet/upernet_eva02_large_24_512_slide_80k.py \
      eva02_L_ade_seg_upernet_sz512.pth \
      --show-dir OUTPUT_DIR \
      --opacity 1
   ```

6. Convert the segmentation maps with better color mapping for classes commonly seen in bedrooms.

```shell
python tools/color2cls.py INPUT_DIR OUTPUT_DIR --pal 1 --ext png
python tools/cls2color.py INPUT_DIR OUTPUT_DIR --pal 2
```

</details>

Then collect the segmentation maps in lmdb.

```shell
python tools/makedb.py INPUT_DIR OUTPUT_DB
```

## Training

To reproduce the trained models, the following script is the basic setting for all tasks. The script is adaped from an [example](https://github.com/huggingface/diffusers/blob/v0.20.0/examples/text_to_image/train_text_to_image_lora.py) provided by huggingface.

```shell
DATA_DIR="/path/to/data"
TARGET_DB="/path/to/target"
OUTPUT_DIR="/path/to/output"

accelerate launch --mixed_precision="fp16" train.py \
    --train_data_dir=$DATA_DIR \
    --train_batch_size=8 \
    --max_train_steps=50000 \
    --learning_rate=1e-04 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=0 \
    --output_dir=$OUTPUT_DIR \
    --target_db=$TARGET_DB \
    --prediction_type="v_prediction"
```

Additionally, for **depths**, set `--target_mode=F` and `--target_scale=8`.

For **depths**, **albedo**, **shading**, and **segmentation**, set `--random_flip`.

For **albedo**, set `--target_extra_key=r`.

For **shading**, set `--target_extra_key=s`.

To add and train lora for only self-attention, set `--self_attn_only`.

To disable prompts, set `--disable_prompts`.

To enable xformers, set `--enable_xformers_memory_efficient_attention`.

## Inference

To generate predictions, run `infer.py` with the same options you run `train.py`.

```shell
DATA_DIR="/path/to/source/images" # optinoal
PROMPTS="/path/to/prompts.txt" # optional
LORA_DIR="/path/to/train/output"
OUTPUT_DIR="/path/to/output"

python infer.py \
    --src $DATA_DIR \
    --prompts $PROMPTS \
    --lora-ckpt $LORA_DIR \
    --output $OUTPUT_DIR \
    --config config.yaml \
    --batch-size 4
```

For **depths**, set `--target-mode=F`, `--target-scale=8`. It generates depths and saves in numpy compressed `npz` format with key `x`.

Optionally set `--target-pred-type`, `--self-attn-only`, and `--disable-prompts` that aligns training. If you don't provide `--src`, it will generate images with the original (no lora) model from `--prompts` . If you don't set `--disable-prompts` but forget to provide `--prompts`, it will raise an error.

More settings for the generation process such as the number of generation steps and guidance scales are in `config.yaml`.

Besides, in the paper we construct the samples of previous diffusion steps with input images and estimated output predictions, but we empirically found using the orignial DDIM, which estimates both input images and output predictions, gives slightly worse in-domain performance but slightly better generalizability. The difference is little, though. The results in the paper were generated by the original DDIM. Set `--use-oracle-ddim` to use exactly the same generation process of the paper.

Also note that the words in the options are connected by *hyphens* `-`, not *underscores* `_`.

## Evaluation

The evaluation script runs on GPU. For **normals**,

```shell
python test/eval.py PRED GROUND_TRUTH --metrics l1 angular
```

For **depths**,

```shell
python test/eval.py PRED GROUND_TRUTH --metrics rel delta --ext npz --abs
python test/eval.py PRED GROUND_TRUTH --metrics rmse --ext npz --abs --norm
```

For **albedo** and **shading**,

```shell
python test/eval.py PRED GROUND_TRUTH --metrics mse
```

For **segmentation**, turn output images into class maps.

```shell
python tools/color2cls.py INPUT_DIR OUTPUT_DIR --pal 2 --ext npy --filter
```

Then calculate miou.

```shell
python test/miou.py PRED GROUND_TRUTH
```

The mIoU evaluation is borrowed from [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/v0.30.0/mmseg/core/evaluation/metrics.py).

## Acknowledgement

This repo contains the code from [diffusers](https://github.com/huggingface/diffusers) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

## Citation

```bibtex
@InProceedings{lee2024dmp,
  author    = {Lee, Hsin-Ying and Tseng, Hung-Yu and Lee, Hsin-Ying and Yang, Ming-Hsuan},
  title     = {Exploiting Diffusion Prior for Generalizable Dense Prediction},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
}
```
