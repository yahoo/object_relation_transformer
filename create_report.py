#!/usr/bin/env python2
# Usage example:
# ./create_report.py --help
# ./create_report.py --pickle eval_results/fc_transformer_bu_adaptive_test_coco_eval.pkl --out_dir report_dir_test
#
# After creating the report by using the commands above, you can set up the
# server to serve it using something like the following commands.
# cd report_dir_test
# python -m SimpleHTTPServer 8888
import argparse
from six.moves import cPickle as pickle
from misc.report import create_report


class Args:
    OUT_DIR = 'out_dir'
    COCO_EVAL_PICKLE = 'pickle'


def main():
    args = _get_command_line_arguments()
    out_dir = args[Args.OUT_DIR]
    pickle_path = args[Args.COCO_EVAL_PICKLE]
    coco_eval = _read_pickle(pickle_path)
    create_report(coco_eval, out_dir)


def _get_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--' + Args.COCO_EVAL_PICKLE,
        help='Pickle file with the COCOEvalCap object', required=True)
    parser.add_argument(
        '--' + Args.OUT_DIR, help='Output directory', required=True)
    args_dict = vars(parser.parse_args())
    return args_dict


def _read_pickle(pickle_path):
    with open(pickle_path, 'rb') as pickle_file:
        coco_eval = pickle.load(pickle_file)
    return coco_eval


if __name__ == '__main__':
    main()
