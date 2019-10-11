######################
# The commands below were used to generate results via evaluation as
# well as corresponding reports. The first set of commands can be used
# for the comparative analysis of the Object Relation Transformer
# against the Baseline Transformer. They a beam size of 2 and do not
# use models trained with reinforcement learning. The second set of
# commands can be used to generate the result of Object Relation
# Transformer trained with reinforcement learning, as well as a
# corresponding report.
######################

###################
# First, run eval.py to generate results for Object Relation
# Transformer and Baseline Transformer with beam size 2 on test data
# (these models were not trained with reinforcement learning).
###################

python eval.py --dump_images 0 --model /mnt/efs/home/sherdade/experiments/captioning/transformer_bs15_noamopt10000/model-best.pth --infos_path /mnt/efs/home/sherdade/experiments/captioning/transformer_bs15_noamopt10000/infos_fc_transformer_bu_adaptive-best.pkl  --image_root /mnt/bigdisk/captioning/data/coco/  --input_json /mnt/bigdisk/captioning/data/cocotalk.json --input_fc_dir /mnt/bigdisk/captioning/data/cocobu_adaptive_fc  --input_att_dir /mnt/bigdisk/captioning/data/cocobu_adaptive_att --input_box_dir /mnt/bigdisk/captioning/data/cocobu_adaptive_box --input_rel_box_dir=/mnt/bigdisk/captioning/data/cocobu_adaptive_box_relative --input_label_h5 /mnt/bigdisk/captioning/data/cocotalk_label.h5  --language_eval=1  --beam_size=2  --split=test
mv eval_results eval_results_transformer_bs15_noamopt10000_beam_size_2_split_test

python eval.py --dump_images 0 --model /mnt/efs/home/sherdade/experiments/captioning/relation_bs15_noamopt10000_no_residue/model-best.pth --infos_path /mnt/efs/home/sherdade/experiments/captioning/relation_bs15_noamopt10000_no_residue/infos_fc_transformer_bu_adaptive-best.pkl --image_root /mnt/bigdisk/captioning/data/coco/  --input_json /mnt/bigdisk/captioning/data/cocotalk.json --input_fc_dir /mnt/bigdisk/captioning/data/cocobu_adaptive_fc --input_att_dir /mnt/bigdisk/captioning/data/cocobu_adaptive_att --input_box_dir /mnt/bigdisk/captioning/data/cocobu_adaptive_box --input_rel_box_dir=/mnt/bigdisk/captioning/data/cocobu_adaptive_box_relative/ --input_label_h5 /mnt/bigdisk/captioning/data/cocotalk_label.h5  --language_eval=1  --beam_size=2  --split=test
mv eval_results eval_results_relation_bs15_noamopt10000_no_residue_beam_size_2_split_test

# Create report for the results above (Object Relation Transformer vs
# Baseline Transformer):

./create_report.py --pickle eval_results_best_may_21/eval_results_relation_bs15_noamopt10000_no_residue_beam_size_2_split_test_fixed/fc_transformer_bu_adaptive_test_report_data.pkl   eval_results_best_may_21/eval_results_transformer_bs15_noamopt10000_beam_size_2_split_test_fixed/fc_transformer_bu_adaptive_test_report_data.pkl  --run_names     relation  transformer  --out_dir reports/relation_bs15_noamopt10000_no_residue_vs_transformer_bs15_noamopt10000_beam_size_2_split_test_best_fixed

###################
# Run eval.py with beam size 5 for model trained with reinforcement
# learning. This corresponds to our best run in the paper.
###################

python eval.py --dump_images 0 --model  /mnt/efs/home/sherdade/experiments/captioning/rl_relation_noamopt_false_2000/model-best.pth --infos_path /mnt/efs/home/sherdade/experiments/captioning/rl_relation_noamopt_false_2000/infos_fc_transformer_rl_bu_adaptive-best.pkl  --image_root /mnt/bigdisk/captioning/data/coco/  --input_json /mnt/bigdisk/captioning/data/cocotalk.json --input_fc_dir /mnt/bigdisk/captioning/data/cocobu_adaptive_fc  --input_att_dir /mnt/bigdisk/captioning/data/cocobu_adaptive_att --input_box_dir /mnt/bigdisk/captioning/data/cocobu_adaptive_box --input_rel_box_dir=/mnt/bigdisk/captioning/data/cocobu_adaptive_box_relative --input_label_h5 /mnt/bigdisk/captioning/data/cocotalk_label.h5  --language_eval=1  --beam_size=5  --split=test
mv eval_results  evaluate_results_rl_beam_size_5

# Create correspoding report for run, which was used in order to
# search for and explore failure cases.

./create_report.py --pickle eval_results_rl_beam_size_5/fc_transformer_rl_bu_adaptive_test_report_data.pkl   --run_names     relation_rl_beam_5  --out_dir reports/rl_relation_noamopt_false_2000_best_beam_5_test
