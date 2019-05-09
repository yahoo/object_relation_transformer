#!/usr/bin/env python2
# Usage example:
# ./create_report.py --help
# ./create_report.py --pickle eval_results/fc_transformer_bu_adaptive_test_coco_eval.pkl --out_dir reports/test_dir
#
# After creating the report by using the commands above, you can set up the
# server to serve it using something like the following commands.
# cd reports/
# python -m SimpleHTTPServer 8888
import argparse
from six.moves import cPickle as pickle
from misc.report import create_report, ReportConfig
from datetime import datetime


class Args:
    BASE_OUT_DIR = 'out_dir'
    COCO_EVAL_PICKLE = 'pickle'
    ADD_TIME = 'add_time'
    NO_ADD_TIME = 'no_add_time'


def main():
    args = _get_command_line_arguments()
    base_out_dir = args[Args.BASE_OUT_DIR]
    add_time = args[Args.ADD_TIME]
    out_dir = _get_out_dir(base_out_dir, add_time)
    pickle_path = args[Args.COCO_EVAL_PICKLE]
    coco_eval = _read_pickle(pickle_path)
    create_report(coco_eval, ReportConfig(out_dir))


def _get_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--' + Args.COCO_EVAL_PICKLE,
        help='Pickle file with the COCOEvalCap object', required=True)
    parser.add_argument(
        '--' + Args.BASE_OUT_DIR, help='Output directory', required=True)
    parser.add_argument(
        '--' + Args.ADD_TIME, help='Add a timestamp to the output directory',
        default=False, required=False, action='store_true', dest=Args.ADD_TIME)
    parser.add_argument(
        '--' + Args.NO_ADD_TIME,
        help='Don\'t add a timestamp to the output directory',
        required=False, action='store_false', dest=Args.ADD_TIME)
    args_dict = vars(parser.parse_args())
    return args_dict


def _get_out_dir(base_out_dir, add_time):
    if add_time:
        date_string = datetime.now().strftime('%Y-%m-%d--%H_%M_%S')
        out_dir = "%s_%s" % (base_out_dir, date_string)
    else:
        out_dir = base_out_dir
    return out_dir


def _read_pickle(pickle_path):
    with open(pickle_path, 'rb') as pickle_file:
        coco_eval = pickle.load(pickle_file)
    return coco_eval


if __name__ == '__main__':
    main()
