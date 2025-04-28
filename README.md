# CTRL-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion

This repo is the official implementation for _CTRL-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion_.

**[Kai He](http://academic.hekai.site/), [Chin-Hsuan Wu](https://chinhsuanwu.github.io/), [Igor Gilitschenski](https://tisl.cs.toronto.edu/author/igor-gilitschenski/).**

**CVPR 2025**

**[[Project Page]](https://ihe-kaii.github.io/CTRL-D/)** **[[Paper Link]](https://arxiv.org/abs/2412.01792)**

![teaser](./imgs/teaser.png)

## Abstract

Recent advances in 3D representations, such as Neural Radiance Fields and 3D Gaussian Splatting, have greatly improved realistic scene modeling and novel-view synthesis. However, achieving controllable and consistent editing in dynamic 3D scenes remains a significant challenge. Previous work is largely constrained by its editing backbones, resulting in inconsistent edits and limited controllability. In our work, we introduce a novel framework that first fine-tunes the InstructPix2Pix model, followed by a two-stage optimization of the scene based on deformable 3D Gaussians. Our fine-tuning enables the model to "learn" the editing ability from a single edited reference image, transforming the complex task of dynamic scene editing into a simple 2D image editing process. By directly learning editing regions and styles from the reference, our approach enables consistent and precise local edits without the need for tracking desired editing regions, effectively addressing key challenges in dynamic scene editing. Then, our two-stage optimization progressively edits the trained dynamic scene, using a designed edited image buffer to accelerate convergence and improve temporal consistency. Compared to state-of-the-art methods, our approach offers more flexible and controllable local scene editing, achieving high-quality and consistent results.

## Installation

```shell
conda create -n CTRL_D python=3.8
conda activate CTRL_D

# [Optional] install CUDA 11.7
conda install cuda -c nvidia/label/cuda-11.7.0

# install pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# install dependencies
pip install -r requirements_monocular.txt # for monocular scenes
pip install -r requirements_multiview.txt # for multi-view scenes
```

## Running

### Data Preparation

- **Monocular scenes.** You can download scenes from [DyCheck Dataset](https://drive.google.com/drive/folders/1cBw3CUKu2sWQfc_1LbFZGbpdQyTFzDEX).
- **Multi-view scenes.** You can download scenes from [DyNeRF Dataset](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0).

### Quick start

We take the *mochi-high-five* scene from DyCheck dataset as an example. You can train the scene from the scratch or download our pre-trained dynamic 3D Gaussian scenes from [here](https://huggingface.co/IHe-KaiI/CTRL-D/tree/main). 

Update all paths in ```run.sh``` with local paths in your system.

- Path to the 2D editing results (`"EDIT_DIR"`).
- Path to the data for scene reconstruction (`"DATA_DIR"`).
- Path to the rgb images in the data ("`IMAGE_DATA_DIR`").
- Path to the output folder (`"OUTPUT_DIR"`).

Then, run

```
bash run.sh
```

### Personalization

Prepare a pair of images with source image and edited image, and make sure name your editing pair like `${NAME}.png` (edited image) and `${NAME}_src.png` (source image).

Then run the personalization with:

```shell
python src/personalization.py \
  --pretrained_model_name_or_path "timbrooks/instruct-pix2pix"  \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir "${EDIT_DIR}" \
  --instance_prompt "${SPECIAL_PROMPT}" \
  --class_data_dir "${OUTPUT_DIR}/class_samples" \
  --class_prompt "${PROMPT}" \
  --validation_prompt "${SPECIAL_PROMPT}" \
  --output_dir "${OUTPUT_DIR}/personalization" \
  --image_root="${IMAGE_DATA_DIR}" \
  --validation_images 0_00000.png \
    0_00050.png  \
    0_00100.png  \
  --max_train_steps 4000 \
  --resolution 384
```

- The `${SPECIAL_PROMPT}` is adding a special token `<kth>` on top of the `${PROMPT}`. For example, 

  ```
  PROMPT="Turn the cat into a fox"
  SPECIAL_PROMPT="Turn the cat into a <kth> fox"
  ```

### Optimization

**For monocular scenes:**

```shell
python src/Deformable-3D-Gaussians/train.py \
    -s "${DATA_DIR}" \
    -m "${OUTPUT_DIR}" \
    --iterations 30000 \
    --edit_iterations 30000 \
    --diffusion_model_checkpoint "${PATH_TO_CHECKPOINT}" \
    --prompt "${SPECIAL_PROMPT}" \
    --idx "${INDEX_OF_KEYFRAME}"
```

- If you already have a pre-trained scene, you can load the checkpoint with `--load_checkpoint ${ITERATION}`.
- To perform the stage 1 of the optimization in the paper, we optimize the 3D Gaussians with only the
  edited keyframe, so that we assign the index of the keyframe using `--idx`.

**For multi-view scenes:**

```shell
# Step 1. for the original scene optimization
python src/4DGaussians/train.py \
    -s "${DATA_DIR}" \
    --expname "${EXP_NAME}" \
    --configs "${PATH_TO_CONFIG}" \
    --iterations 14000 \
# Step 2. for editing the scene
python src/4DGaussians/train.py \
    -s "${DATA_DIR}" \
    --expname "${EXP_NAME}" \
    --configs "${PATH_TO_CONFIG}" \
    --start_checkpoint 14000 \
    --edit_iterations 6000 \
    --diffusion_model_checkpoint "${PATH_TO_CHECKPOINT}" \
    --prompt "${SPECIAL_PROMPT}" \
    --idx "${INDEX_OF_KEYFRAME}"
```

- For example, `EXP_NAME=dynerf/sear_steak`, `PATH_TO_CONFIG=./src/4DGaussians/arguments/dynerf/sear_steak.py`.
- If you already have a pre-trained scene, you can skip the step 1.

### Visualization

**For monocular scenes:**

```shell
python src/Deformable-3D-Gaussians/render.py -m "${OUTPUT_DIR}"
```

**For multi-view scenes:**

```shell
python src/4DGaussians/render.py --model_path "output/${EXP_NAME}"  --skip_train --configs "${PATH_TO_CONFIG}"
```

**Tips:**

- Make sure the personalized InstructPix2Pix and the pre-trained 3D Gaussians are good. Sometimes, the best personalized model is not the latest one; pick one with the best visualization in validation.
- Normally, optimization with $20000$ to $30000$ iterations for monocular scenes and $6000$ to $10000$ for multi-view scenes is enough.

## Acknowledgment

This project is built upon **[Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)**, **[4DGaussians](https://github.com/hustvl/4DGaussians)**, and **[TIP-Editor](https://github.com/zjy526223908/TIP-Editor)**, and refer some parallel ideas from **[Instruct-4D-to-4D](https://github.com/Friedrich-M/Instruct-4D-to-4D)**. We thank all the authors for their impressive repos.

## Citation

```
@article{he2024ctrl,
  title={CTRL-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion},
  author={He, Kai and Wu, Chin-Hsuan and Gilitschenski, Igor},
  journal={arXiv preprint arXiv:2412.01792},
  year={2024}
}
```

