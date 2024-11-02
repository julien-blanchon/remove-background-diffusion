accelerate launch src/train.py \
  --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --modality "background" \
  --noise_type "zeros" \
  --max_train_steps 20000 \
  --checkpointing_steps 20000 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing \
  --learning_rate 3e-05 \
  --lr_total_iter_length 20000 \
  --lr_exp_warmup_steps 100 \
  --mixed_precision "no" \
  --output_dir "runs/stable_diffusion_e2e_ft_background" \
  --enable_xformers_memory_efficient_attention \
  "$@";