## Eval
Run `eval.py` on a trained model checkpoint saved under `--output_dir`: `model.pth`, or `model-best.pth`.

You can specify the following parameters:

* `model` (`str`): path to trained model checkpoint.
* `image_root` (`str`): path to base_dir of your validation data.
* `num_images` (`int`): if set to -1 runs on all images.
* `dump_images` (`int`): if set to 1, a sample of image captions is dumped to the vis dir. If set to 0 nothing happens.
* `language_eval` (`int`): if set to 1, it outputs accuracy metrics ('CIDEr, BLEU, ROUGE, METEOR, SPICE).

example:
```bash 
python eval.py \
        --dump_images 1 \
        --num_images 100 \
        --model /efs/home/sherdade/experiments/captioning/relation_rewritten_with_relu/model.pth \
        --infos_path /efs/home/sherdade/experiments/captioning/relation_rewritten_with_relu/infos_fc_transformer_bu_adaptive-best.pkl \
        --image_root /mydisk/data/captioning_data/coco/ \
        --input_json /mydisk/data/captioning_data/cocotalk.json \
        --input_fc_dir /mydisk/data/captioning_data/cocobu_adaptive_fc \
        --input_att_dir /mydisk/data/captioning_data/cocobu_adaptive_att \
        --input_box_dir /mydisk/data/captioning_data/cocobu_adaptive_box \
        --input_rel_box_dir=/mydisk/data/captioning_data/cocobu_adaptive_box_relative/ \
        --input_label_h5 /mydisk/data/captioning_data/cocotalk_label.h5  \
        --use_box 1
        --language_eval 1
```

## Inference
For inference set `--language_eval=0`.

## Visualize sample
To visualize a sample of generated captions:
1. Run eval.py with flag --dump_images 1.
2. Start a server to display sample images.
```bash
$ cd vis
$ python -m SimpleHTTPServer
```
