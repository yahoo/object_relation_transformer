## Download COCO captions and preprocess
```
mkdir data
cd data
wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip
cd ..
python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

## Download ResNet Weights
*Download weights from [https://drive.google.com/drive/folders/0B7fNdx_jAqhtbVYzOURMdDNHSGM](here).
*Copy to data/imagenet_weights (folder should be created)

## Get COCO data
cd /mnt/bigdisk/data
```
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
cd ~/workspace
wget https://msvocds.blob.core.windows.net/images/262993_z.jpg
mv 262993_z.jpg /mnt/data/bigdisk/coco/train2014/COCO_train2014_000000167126.jpg
```

## Pre-process COCO data
```
python scripts/prepro_feats.py  --input_json data/dataset_coco.json  --output_dir /mnt/bigdisk/data/cocotalk --images_root /mnt/bigdisk/data/coco/
```
Note: you need to add the current directory to your PYTHONPATH to get this to work

## Train on COCO with ResNet101 features
```
python train.py --id fc --caption_model fc --input_json data/cocotalk.json --input_fc_dir /mnt/bigdisk/data/cocotalk_fc --input_att_dir /mnt/bigdisk/data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path expt001-resnet101-features-fc-model  --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30  --language_eval 1
```
Note: need to make shared memory writeable as suggested by [this](https://stackoverflow.com/questions/2009278/python-multiprocessing-permission-denied)
Note: this appears to only work with a single GPU (export CUDA_VISIBLE_DEVICES="0")

## Generate bottom-up features (adaptive)
```
pushd .
cd /mnt/bigdisk/data
mkdir bu_data
cd bu_data
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip
unzip trainval.zip
popd
python scripts/make_bu_data.py --downloaded_feats /mnt/bigdisk/data/bu_data --output_dir /mnt/bigdisk/data/cocobu_adaptive
```

## Generate bottom-up features (fixed)
```
pushd .
cd /mnt/bigdisk/data/bu_data
https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
unzip trainval_36.zip
popd
python scripts/make_bu_data.py --downloaded_feats /mnt/bigdisk/data/bu_data --output_dir /mnt/bigdisk/data/cocobu_adaptive
```
Note: the script needs to be modified to load the appropriate files


## Train on COCO with bottom-up features (fixed)
```
python train.py --id fc_bu_fixed --caption_model fc --input_json data/cocotalk.json --input_fc_dir /mnt/bigdisk/data/cocobu_fc --input_att_dir /mnt/bigdisk/data/cocobu_att --input_box_dir /mnt/bigdisk/data/cocobu_box --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path experiments/expt002-bottom-up-features-fc-model  --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30  --language_eval 1
```

## Train on COCO with ResNet101 features and self-critical training
```
python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl 2data/coco-train --split train
cd experiments
mkdir expt003-resnet101-features-fc-rl-model
cp -r expt001-resnet101-features-fc-model/* expt003-resnet101-features-fc-rl-model
cd expt003-resnet101-features-fc-rl-model
mv infos_fc-best.pkl  infos_fc_rl-best.pkl
mv infos_fc.pkl infos_fc_rl.pkl
cd ../..
python train.py --id fc_rl --caption_model fc --input_json data/cocotalk.json --input_fc_dir /mnt/bigdisk/data/cocotalk_fc --input_att_dir /mnt/bigdisk/data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-5 --start_from experiments/expt003-resnet101-features-fc-rl-model --checkpoint_path experiments/expt003-resnet101-features-fc-rl-model --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --self_critical_after 30
```

## Train on COCO with ResNet101 features and transformer
```
python train.py --id fc_transformer --caption_model transformer --noamopt --noamopt_warmup 20000 --label_smoothing 0.0 --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_fc_dir /mnt/bigdisk/data/cocotalk_fc --input_att_dir /mnt/bigdisk/data/cocotalk_att --batch_size 10 --beam_size 1 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path experiments/expt004-resnet101-features-transformer-model --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --max_epochs 30
```

## Train on COCO with bottom-up features and transformer
```
python train.py --id fc_transformer_bu --caption_model transformer --noamopt --noamopt_warmup 20000 --label_smoothing 0.0 --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_fc_dir /mnt/bigdisk/data/cocobu_fc --input_att_dir /mnt/bigdisk/data/cocobu_att --input_box_dir /mnt/bigdisk/data/cocobu_box --batch_size 10 --beam_size 1 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path experiments/expt005-bottom-up-features-transformer-model --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --max_epochs 30
```

## Train on COCO with bottom-up features (adaptive)
```
python train.py --id fc_bu_adaptive --caption_model fc --input_json data/cocotalk.json --input_fc_dir /mnt/bigdisk/data/cocobu_adaptive_fc --input_att_dir /mnt/bigdisk/data/cocobu_adaptive_att --input_box_dir /mnt/bigdisk/data/cocobu_adaptive_box --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path experiments/expt006-adaptive-bottom-up-features-fc-model  --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30  --language_eval 1
```
## Train on COCO with bottom-up features and transformer (adaptive)
```
python train.py --id fc_transformer_bu_adaptive --caption_model transformer --noamopt --noamopt_warmup 20000 --label_smoothing 0.0 --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_fc_dir /mnt/bigdisk/data/cocobu_adaptive_fc --input_att_dir /mnt/bigdisk/data/cocobu_adaptive_att --input_box_dir /mnt/bigdisk/data/cocobu_adaptive_box --batch_size 10 --beam_size 1 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path experiments/expt007-adaptive-bottom-up-features-transformer-model --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --max_epochs 30
```


## Train on COCO with bottom-up features (adaptive) and self-critical training
```
cd experiments
mkdir expt008-adaptive-bottom-up-features-fc-rl-model
cp expt006-adaptive-bottom-up-features-fc-model/* expt008-adaptive-bottom-up-features-fc-rl-model
cd expt008-adaptive-bottom-up-features-fc-rl-model
mv infos_fc_bu_adaptive-best.pkl  infos_fc_rl_bu_adaptive-best.pkl
mv infos_fc_bu_adaptive.pkl infos_fc_rl_bu_adaptive.pkl
cd ../..
python train.py --id fc_rl_bu_adaptive --caption_model fc --input_json data/cocotalk.json --input_fc_dir /mnt/bigdisk/data/cocobu_adaptive_fc --input_att_dir /mnt/bigdisk/data/cocobu_adaptive_att --input_box_dir /mnt/bigdisk/data/cocobu_adaptive_box --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --start_from experiments/expt008-adaptive-bottom-up-features-fc-rl-model --checkpoint_path experiments/expt008-adaptive-bottom-up-features-fc-rl-model  --save_checkpoint_every 6000 --val_images_use 5000  --language_eval 1 --self_critical_after 30 --max_epochs 60
```

## Train on COCO with bottom-up features (adaptive), transformer, and self-critical training
```
cd experiments
mkdir expt009-adaptive-bottom-up-features-transformer-rl-model
cp expt007-bottom-up-features-adaptive-transformer-model/* expt009-adaptive-bottom-up-features-transformer-rl-model
cd expt009-adaptive-bottom-up-features-transformer-rl-model
mv infos_fc_transformer_bu_adaptive-best.pkl  infos_fc_transformer_rl_bu_adaptive-best.pkl
mv infos_fc_transformer_bu_adaptive.pkl infos_fc_transfomer_rl_bu_adaptive.pkl
cd ../..
python train.py --id fc_transformer_rl_bu_adaptive --caption_model transformer --input_json data/cocotalk.json --input_fc_dir /mnt/bigdisk/data/cocobu_adaptive_fc --input_att_dir /mnt/bigdisk/data/cocobu_adaptive_att --input_box_dir /mnt/bigdisk/data/cocobu_adaptive_box --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --beam_size 1 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --start_from experiments/expt009-adaptive-bottom-up-features-transformer-rl-model --checkpoint_path experiments/expt009-adaptive-bottom-up-features-transformer-rl-model  --save_checkpoint_every 6000 --val_images_use 5000  --language_eval 1 --self_critical_after 30 --max_epochs 60
```
