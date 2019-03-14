## Eval
To run evaluation, run `eval.py` with arguments
```bash
--model (path to trained model checkpoint)
--image_root (base_dir of your val_data)
--num_images  (-1 runs on all images)
--dump_images  (if 1 sample of images and captions are dumped to the vis directory. if 0 not.)
```
example:
```bash 
python eval.py --dump_images 1 --num_images 5000 --model /efs/home/sherdade/experiments/captioning/simao-adaptive-bottom-up-features-transformer-model/model-best.pth --infos_path /efs/home/sherdade/experiments/captioning/simao-adaptive-bottom-up-features-transformer-model/infos_fc_transformer_bu_adaptive-best.pkl --language_eval 1 --image_root /mydisk/Data/captioning_data/coco/
```

## Visualize sample
1. Run eval.py with flag --dump_images 1.
2. Start a server to display sample images.
```bash
$ cd vis
$ python -m SimpleHTTPServer
```
