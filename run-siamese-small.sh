export BS=12; rm -r output_dir; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../src USE_TF=0 deepspeed  \
./finetune_trainer.py --model_name_or_path "./templates/siamese-t5-small-template" --output_dir output_dir --adam_eps 1e-06 \
--data_dir qasc --do_eval --do_predict --do_train --evaluation_strategy=steps  --freeze_embeds --label_smoothing 0.1 \
--learning_rate 3e-5 --logging_first_step --logging_steps 50 --max_source_length 100 --max_target_length 70 \
--num_train_epochs 2 --overwrite_output_dir --per_device_eval_batch_size $BS --per_device_train_batch_size $BS \
--eval_steps 25000 --label_smoothing_factor 0 --save_steps 3 --src_lang en_XX --task translation \
--test_max_target_length 70 --tgt_lang ro_RO --val_max_target_length 70 --warmup_steps 500 --n_train 40000 --n_val 2000 \
--n_test 2000 --fp16 --deepspeed ds_config.json
# google/t5-v1_1-xl    BS=14
# t5-3b    BS=12
#  --dataloader_num_workers 1