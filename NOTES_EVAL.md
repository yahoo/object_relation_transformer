## One Model vs Ensemble
After training, you will have two checkpoints under `--output_dir`: `model.pth`, and `model-best.pth`.

`model-best.pth` is an ensemble of several training checkpoints.

## Eval
To run evaluation, run `eval.py` with arguments


* `model` (`str`): path to trained model checkpoint.
* `image_root` (`str`): path to base_dir of your validation data.
* `num_images` (`int`): if set to -1 runs on all images.
* `dump_images` (`int`): if set to 1, a sample of image captions is dumped to the vis dir. If set to 0 nothing happens.

example:
```bash 
python eval.py --dump_images 1 --num_images 100 --model /efs/home/sherdade/experiments/captioning/simao-adaptive-bottom-up-features-transformer-model/model-best.pth --infos_path /efs/home/sherdade/experiments/captioning/simao-adaptive-bottom-up-features-transformer-model/infos_fc_transformer_bu_adaptive-best.pkl --language_eval 1 --image_root /mydisk/Data/captioning_data/coco/
```

## Visualize sample
1. Run eval.py with flag --dump_images 1.
2. Start a server to display sample images.
```bash
$ cd vis
$ python -m SimpleHTTPServer
```
