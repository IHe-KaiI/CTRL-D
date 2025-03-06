EDIT_DIR=./data/mochi-high-five-fox
DATA_DIR=./data/mochi-high-five
IMAGE_DATA_DIR=./data/mochi-high-five/rgb/1x
OUTPUT_DIR=./output/mochi-high-five-fox
PROMPT="Turn the cat into a fox"
SPECIAL_PROMPT="Turn the cat into a <kth> fox"

### Fine-tune the InstructPix2Pix
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

### Load the pre-trained scene and edit the scene with the personalized InstructPix2Pix.
python src/Deformable-3D-Gaussians/train.py \
  -s "${DATA_DIR}" \
  -m "${OUTPUT_DIR}" \
  --iterations 30000 \
  --load_checkpoint 30000 \
  --edit_iterations 20000 \
  --diffusion_model_checkpoint "${OUTPUT_DIR}/scene_personalization/checkpoint-4000" \
  --prompt "${SPECIAL_PROMPT}" \
  --idx 42

### Render the results
python src/Deformable-3D-Gaussians/render.py -m "${OUTPUT_DIR}"
