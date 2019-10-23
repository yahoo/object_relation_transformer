##########################################################
# Copyright 2019 Oath Inc.
# Licensed under the terms of the MIT license.
# Please see LICENSE file in the project root for terms.
##########################################################
# The commands below should be used to run inference for different
# models on the MSCOCO dataset, as well as to generate the
# corresponding HTML reports containing the results and the
# comparative analysis between methods. In particular, they can be
# used in order to reproduce the results from tables 4 and 5 in the
# NeurIPS 2019 paper describing the Object Relation Transformer.
#
# The first set of commands can be used for the comparative analysis
# of the Object Relation Transformer against the Baseline
# Transformer. These commands use a beam size of 2 and do not use
# models trained with reinforcement learning. The second set of
# commands can be used to generate the result of Object Relation
# Transformer trained with reinforcement learning, as well as a
# corresponding report. This corresponds to the best run that was
# presented in the NeurIPS paper.
######################

# Setting the directory with COCO-related data to be used in later
# commands.
COCO_DATA_DIR=/my/coco/data/

###################
# First, run eval.py to generate results for Object Relation
# Transformer and Baseline Transformer with beam size 2 on test data
# (these models were not trained with reinforcement learning).
###################
BASELINE_TRANSFORMER_MODEL_DIR=/my/models/baseline_transformer
BASELINE_TRANSFORMER_RESULTS_DIR=/my/results/baseline_transformer_beam_size_2_split_test
python eval.py --dump_images 0 --model ${BASELINE_TRANSFORMER_MODEL_DIR}/model-best.pth --infos_path ${BASELINE_TRANSFORMER_MODEL_DIR}/infos_fc_transformer_bu_adaptive-best.pkl  --image_root ${COCO_DATA_DIR}/coco/  --input_json ${COCO_DATA_DIR}/cocotalk.json --input_fc_dir ${COCO_DATA_DIR}/cocobu_adaptive_fc  --input_att_dir ${COCO_DATA_DIR}/cocobu_adaptive_att --input_box_dir ${COCO_DATA_DIR}/cocobu_adaptive_box --input_rel_box_dir=${COCO_DATA_DIR}/cocobu_adaptive_box_relative --input_label_h5 ${COCO_DATA_DIR}/cocotalk_label.h5  --language_eval=1  --beam_size=2  --split=test
mv eval_results ${BASELINE_TRANSFORMER_RESULTS_DIR}

RELATION_TRANSFORMER_MODEL_DIR=/my/models/relation_transformer
RELATION_TRANSFORMER_RESULTS_DIR=/my/results/relation_transformer_beam_size_2_split_test
python eval.py --dump_images 0 --model ${RELATION_TRANSFORMER_MODEL_DIR}/model-best.pth --infos_path ${RELATION_TRANSFORMER_MODEL_DIR}/infos_fc_transformer_bu_adaptive-best.pkl --image_root ${COCO_DATA_DIR}/coco/  --input_json ${COCO_DATA_DIR}/cocotalk.json --input_fc_dir ${COCO_DATA_DIR}/cocobu_adaptive_fc --input_att_dir ${COCO_DATA_DIR}/cocobu_adaptive_att --input_box_dir ${COCO_DATA_DIR}/cocobu_adaptive_box --input_rel_box_dir=${COCO_DATA_DIR}/cocobu_adaptive_box_relative/ --input_label_h5 ${COCO_DATA_DIR}/cocotalk_label.h5  --language_eval=1  --beam_size=2  --split=test
mv eval_results ${RELATION_TRANSFORMER_RESULTS_DIR}

# Create report for the results above (Object Relation Transformer vs
# Baseline Transformer):
./create_report.py --pickle ${RELATION_TRANSFORMER_RESULTS_DIR}/fc_transformer_bu_adaptive_test_report_data.pkl  ${BASELINE_TRANSFORMER_RESULTS_DIR}/fc_transformer_bu_adaptive_test_report_data.pkl  --run_names   relation  transformer  --out_dir reports/relation_vs_baseline_beam_size_2_split_test

###################
# Run eval.py with beam size 5 for model trained with reinforcement
# learning. This corresponds to our best run in the paper.
###################
RL_RELATION_TRANSFORMER_MODEL_DIR=/my/models/rl_relation_transformer
RL_RELATION_TRANSFORMER_RESULTS_DIR=/my/results/rl_relation_transformer_beam_size_5_split_test
python eval.py --dump_images 0 --model  ${RL_RELATION_TRANSFORMER_MODEL_DIR}/model-best.pth --infos_path ${RL_RELATION_TRANSFORMER_MODEL_DIR}/infos_fc_transformer_rl_bu_adaptive-best.pkl  --image_root ${COCO_DATA_DIR}/coco/  --input_json ${COCO_DATA_DIR}/cocotalk.json --input_fc_dir ${COCO_DATA_DIR}/cocobu_adaptive_fc  --input_att_dir ${COCO_DATA_DIR}/cocobu_adaptive_att --input_box_dir ${COCO_DATA_DIR}/cocobu_adaptive_box --input_rel_box_dir=${COCO_DATA_DIR}/cocobu_adaptive_box_relative --input_label_h5 ${COCO_DATA_DIR}/cocotalk_label.h5  --language_eval=1  --beam_size=5  --split=test
mv eval_results  ${RL_RELATION_TRANSFORMER_RESULTS_DIR}

# Create correspoding report for run, which was used in our paper in
# order to search for and explore failure cases.
./create_report.py --pickle ${RL_RELATION_TRANSFORMER_RESULTS_DIR}/fc_transformer_rl_bu_adaptive_test_report_data.pkl --run_names relation_rl_beam_5 --out_dir reports/rl_relation_transformer_beam_size_5_split_test
