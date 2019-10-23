##########################################################
# Copyright 2019 Oath Inc.
# Licensed under the terms of the MIT license.
# Please see LICENSE file in the project root for terms.
##########################################################
######################################################################
#Up-Down + LSTM
#training 
python train.py --id topdown_bu --caption_model topdown --input_json data/cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_label_h5 data/cocotalk_label.h5 --checkpoint_path log_topdown_bu --batch_size 15 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --max_epochs 30
#evaluation
python eval.py --dump_images 0 --num_images 5000 --model log_topdown_bu/model.pth --infos_path log_topdown_bu/infos_topdown_bu-best.pkl --image_root $IMAGE_ROOT --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5  --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --language_eval 1 --beam_size 1
######################################################################

######################################################################
#Up-Down + Transformer
#training
python train.py --id transformer_bu --caption_model transformer --input_json data/cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_label_h5 data/cocotalk_label.h5 --checkpoint_path log_transformer_bu --noamopt --noamopt_warmup 10000 --batch_size 15 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --max_epochs 30
#evaluation
python eval.py --dump_images 0 --num_images 5000 --model log_transformer_bu/model.pth --infos_path log_transformer_bu/infos_transformer_bu-best.pkl --image_root $IMAGE_ROOT --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5  --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --language_eval 1 --beam_size 1
######################################################################

######################################################################
#Up-Down + Object Relation Transformer
#training
python train.py --id relation_transformer_bu --caption_model relation_transformer --input_json data/cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --input_label_h5 data/cocotalk_label.h5 --checkpoint_path log_relation_transformer_bu --noamopt --noamopt_warmup 10000 --batch_size 15 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --max_epochs 30 --use_box 1
#evaluation
python eval.py --dump_images 0 --num_images 5000 --model log_relation_transformer_bu/model.pth --infos_path log_relation_transformer_bu/infos_relation_transformer_bu-best.pkl --image_root $IMAGE_ROOT --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5  --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --use_box 1 --language_eval 1 --beam_size 1
######################################################################

######################################################################
#Up-Down + Object Relation Transformer + Beamsize 2
#Training is the same as above and evaluation is done
#with beamsize=2
#training
python train.py --id relation_transformer_bu --caption_model relation_transformer --input_json data/cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --input_label_h5 data/cocotalk_label.h5 --checkpoint_path log_relation_transformer_bu --noamopt --noamopt_warmup 10000 --batch_size 15 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --max_epochs 30 --use_box 1
#evaluation 
python eval.py --dump_images 0 --num_images 5000 --model log_relation_transformer_bu/model.pth --infos_path log_relation_transformer_bu/infos_relation_transformer_bu-best.pkl --image_root $IMAGE_ROOT --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5  --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --use_box 1 --language_eval 1 --beam_size 2
######################################################################

######################################################################
#Up-Down + Object Relation Transformer + Self-Critical + Beamsize 5
#training (crossentropy loss, 30 epochs)
python train.py --id relation_transformer_bu --caption_model relation_transformer --input_json data/cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --input_label_h5 data/cocotalk_label.h5 --checkpoint_path log_relation_transformer_bu --noamopt --noamopt_warmup 10000 --batch_size 15 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --max_epochs 30 --use_box 1
#copy model parameters to continue training with self-critical RL
bash scripts/copy_model.sh relation_transformer_bu relation_transformer_bu_rl
#training (self-critical RL, 30 additional epochs)
python train.py --id relation_transformer_bu_rl --caption_model relation_transformer --input_json data/cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_label_h5 data/cocotalk_label.h5  --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --input_label_h5 data/cocotalk_label.h5 --checkpoint_path log_relation_transformer_bu_rl --batch_size 10 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --start_from log_transformer_bu_rl --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --self_critical_after 30 --max_epochs 60 --use_box 1
#evaluation
python eval.py --dump_images 0 --num_images 5000 --model log_relation_transformer_bu_rl/model.pth --infos_path log_relation_transformer_bu_rl/infos_relation_transformer_bu-best.pkl --image_root $IMAGE_ROOT --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5  --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --language_eval 1 --beam_size 5
#######################################################################