import numpy
import os
import sys
import shutil
import matplotlib.pyplot as plt
# By using Agg, pyplot will not try to open a display
plt.switch_backend('Agg')
from matplotlib.figure import Figure
from pandas import DataFrame, Series, concat
from pandas.io.json import json_normalize
from typing import IO, Optional, Dict, List
from scipy.stats import ttest_rel
from six.moves import cPickle as pickle
sys.path.append("coco-caption")

HTML_IMAGE_WIDTH_PIXELS = 400
HTML_IMAGE_ALIGN_LEFT = 'left'
SUMMARY_VALUES_COLUMN_NAME = 'Value'
RESULT_KEY_CAPTION = 'caption'
GROUND_TRUTH_KEY_CAPTION = 'caption'
PREDICTION_KEY_IMAGE_ID = 'image_id'
PREDICTION_KEY_FILE_PATH = 'file_path'
# In COCOEvalCAP.eval, the SPICE metric has a different name than in
# COCOEvalCAP.imgToEval, so we record it here so we can map it to the same
# value later.
COCO_EVAL_SPICE_COLUMN = 'SPICE'


class EvalColumns:
    IMAGE_ID = 'image_id'
    PATH = 'path'
    RESULT_CAPTION = 'resultCaption'
    GROUND_TRUTH_CAPTIONS = 'groundTruthCaptions'
    BLEU1 = 'Bleu_1'
    BLEU2 = 'Bleu_2'
    BLEU3 = 'Bleu_3'
    BLEU4 = 'Bleu_4'
    CIDER = 'CIDEr'
    METEOR = 'METEOR'
    ROUGE_L = 'ROUGE_L'
    SPICE = 'SPICE.All.f'
    SPICE_PR = 'SPICE.All.pr'
    SPICE_RE = 'SPICE.All.re'
    SPICE_OBJECT = 'SPICE.Object.f'
    SPICE_OBJECT_PR = 'SPICE.Object.pr'
    SPICE_OBJECT_RE = 'SPICE.Object.re'
    SPICE_RELATION = 'SPICE.Relation.f'
    SPICE_RELATION_PR = 'SPICE.Relation.pr'
    SPICE_RELATION_RE = 'SPICE.Relation.re'
    SPICE_ATTRIBUTE = 'SPICE.Attribute.f'
    SPICE_ATTRIBUTE_PR = 'SPICE.Attribute.pr'
    SPICE_ATTRIBUTE_RE = 'SPICE.Attribute.re'
    SPICE_COLOR = 'SPICE.Color.f'
    SPICE_COLOR_PR = 'SPICE.Color.pr'
    SPICE_COLOR_RE = 'SPICE.Color.re'
    SPICE_CARDINALITY = 'SPICE.Cardinality.f'
    SPICE_CARDINALITY_PR = 'SPICE.Cardinality.pr'
    SPICE_CARDINALITY_RE = 'SPICE.Cardinality.re'
    SPICE_SIZE = 'SPICE.Size.f'
    SPICE_SIZE_PR = 'SPICE.Size.pr'
    SPICE_SIZE_RE = 'SPICE.Size.re'


INDEX_COLUMN_NAME = EvalColumns.IMAGE_ID
COLUMNS_FOR_HISTOGRAM_NON_SPICE = [
    EvalColumns.BLEU1, EvalColumns.BLEU2, EvalColumns.BLEU3, EvalColumns.BLEU4,
    EvalColumns.CIDER, EvalColumns.METEOR, EvalColumns.ROUGE_L]
COLUMNS_FOR_HISTOGRAM_SPICE = [
    EvalColumns.SPICE, EvalColumns.SPICE_PR, EvalColumns.SPICE_RE,
    EvalColumns.SPICE_OBJECT, EvalColumns.SPICE_OBJECT_PR,
    EvalColumns.SPICE_OBJECT_RE,
    EvalColumns.SPICE_RELATION, EvalColumns.SPICE_RELATION_PR,
    EvalColumns.SPICE_RELATION_RE,
    EvalColumns.SPICE_ATTRIBUTE, EvalColumns.SPICE_ATTRIBUTE_PR,
    EvalColumns.SPICE_ATTRIBUTE_RE,
    EvalColumns.SPICE_COLOR, EvalColumns.SPICE_COLOR_PR,
    EvalColumns.SPICE_COLOR_RE,
    EvalColumns.SPICE_CARDINALITY, EvalColumns.SPICE_CARDINALITY_PR,
    EvalColumns.SPICE_CARDINALITY_RE,
    EvalColumns.SPICE_SIZE, EvalColumns.SPICE_SIZE_PR,
    EvalColumns.SPICE_SIZE_RE]
EXTRA_SUMMARY_COLUMNS = [
    EvalColumns.SPICE_PR, EvalColumns.SPICE_RE,
    EvalColumns.SPICE_OBJECT, EvalColumns.SPICE_OBJECT_PR,
    EvalColumns.SPICE_OBJECT_RE,
    EvalColumns.SPICE_RELATION, EvalColumns.SPICE_RELATION_PR,
    EvalColumns.SPICE_RELATION_RE,
    EvalColumns.SPICE_ATTRIBUTE, EvalColumns.SPICE_ATTRIBUTE_PR,
    EvalColumns.SPICE_ATTRIBUTE_RE,
    EvalColumns.SPICE_COLOR, EvalColumns.SPICE_COLOR_PR,
    EvalColumns.SPICE_COLOR_RE,
    EvalColumns.SPICE_CARDINALITY, EvalColumns.SPICE_CARDINALITY_PR,
    EvalColumns.SPICE_CARDINALITY_RE,
    EvalColumns.SPICE_SIZE, EvalColumns.SPICE_SIZE_PR,
    EvalColumns.SPICE_SIZE_RE]
BASE_SUMMARY_COLUMNS = [
    EvalColumns.BLEU1, EvalColumns.BLEU2, EvalColumns.BLEU3, EvalColumns.BLEU4,
    EvalColumns.CIDER, EvalColumns.METEOR, EvalColumns.ROUGE_L,
    EvalColumns.SPICE]
ALL_SUMMARY_COLUMNS = BASE_SUMMARY_COLUMNS + EXTRA_SUMMARY_COLUMNS


class ReportConfig:

    HISTOGRAM_BINS_DICT = {
        EvalColumns.CIDER: numpy.linspace(0, 6, 6 * 4 + 1),
        EvalColumns.BLEU1: numpy.linspace(0, 1, 21),
        EvalColumns.BLEU2: numpy.linspace(0, 1, 21),
        EvalColumns.BLEU3: numpy.linspace(0, 1, 21),
        EvalColumns.BLEU4: numpy.linspace(0, 1, 21),
        EvalColumns.METEOR: numpy.linspace(0, 1, 21),
        EvalColumns.ROUGE_L: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_PR: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_RE: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_OBJECT: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_OBJECT_PR: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_OBJECT_RE: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_ATTRIBUTE: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_ATTRIBUTE_PR: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_ATTRIBUTE_RE: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_RELATION: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_RELATION_PR: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_RELATION_RE: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_SIZE: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_SIZE_PR: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_SIZE_RE: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_CARDINALITY: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_CARDINALITY_PR: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_CARDINALITY_RE: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_COLOR: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_COLOR_PR: numpy.linspace(0, 1, 21),
        EvalColumns.SPICE_COLOR_RE: numpy.linspace(0, 1, 21)
    }  # type: Dict[str, numpy.ndarray]

    def __init__(self, out_dir):
        # type: (str) -> None
        self.out_dir = out_dir
        self.histogram_bins = ReportConfig.HISTOGRAM_BINS_DICT


class ReportData:

    def __init__(self, coco_eval, predictions, image_root, model_id, split):
        # type (COCOEvalCap, List[Dict[str, Any]], str, str, str) -> None
        self.coco_eval = coco_eval
        self.predictions = predictions
        self.image_root = image_root
        self.model_id = model_id
        self.split = split

    def save_to_pickle(self, pickle_path):
        # type: (str) -> None
        """Save ReportData into a pickle file so that we can create
        visualizations for it later."""
        with open(pickle_path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @staticmethod
    def read_from_pickle(pickle_path, model_id=None):
        # type: (str, Optional[str]) -> ReportData
        with open(pickle_path, 'rb') as pickle_file:
            report_data = pickle.load(pickle_file)
        if model_id:
            report_data.model_id = model_id
        return report_data


class PathForHTML:
    """Keeps track of a regular path, as well as a relative path, which can be
    used to create links in HTML code."""

    def __init__(self, regular, base_dir):
        # type: (str, str) -> None
        self.regular = regular
        self.base_dir = base_dir

    def relative_to(self, other_dir):
        # type: (str) -> str
        return os.path.relpath(self.regular, other_dir)

    def relative(self):
        # type: () -> str
        return self.relative_to(self.base_dir)

    def join(self, to_append):
        # type: (str) -> PathForHTML
        return PathForHTML(os.path.join(self.regular, to_append), self.base_dir)

    def with_base_dir(self, base_dir):
        # type: (str) -> PathForHTML
        return PathForHTML(self.regular, base_dir)


class OutputPaths:

    INDEX_HTML = 'index.html'
    IMAGE_REPORT_DIR_NAME = 'image_reports'
    IMAGE_DIR_NAME = 'images'
    IMAGE_EXTENSION = '.jpg'

    def __init__(self, out_dir):
        # type: (str) -> None
        # Set directories and then run makedirs to create them.
        self.out_dir = out_dir
        self.image_report_dir = os.path.join(
            self.out_dir, OutputPaths.IMAGE_REPORT_DIR_NAME)
        self.image_dir = os.path.join(self.out_dir, OutputPaths.IMAGE_DIR_NAME)
        # For now, just fail if the directory already exists, so we don't
        # overwrite anything accidentally.
        os.makedirs(self.out_dir)
        os.makedirs(self.image_report_dir)
        os.makedirs(self.image_dir)
        # Create some specific relevant paths
        self.report_index_path = os.path.join(self.out_dir,
                                              OutputPaths.INDEX_HTML)


class RunOutputPaths:

    PLOT_DIR_NAME = 'plots'

    def __init__(self, output_paths, run_name):
        # type: (OutputPaths, str) -> None
        self.output_paths = output_paths
        self.run_dir = os.path.join(self.output_paths.out_dir, run_name)
        self.plot_dir = os.path.join(self.run_dir,
                                     RunOutputPaths.PLOT_DIR_NAME)
        os.makedirs(self.run_dir)
        os.makedirs(self.plot_dir)
        self.index_path = os.path.join(self.run_dir, OutputPaths.INDEX_HTML)

    def histogram_image_path(self, column_name):
        # type: (str) -> str
        histogram_file_name = column_name + OutputPaths.IMAGE_EXTENSION
        return os.path.join(self.plot_dir, histogram_file_name)

    def metric_path(self, column_name):
        # type: (str) -> str
        metric_file_name = column_name + '.html'
        return os.path.join(self.run_dir, metric_file_name)


class MetricData:

    def __init__(self, data_frame, column_name, bins):
        # type: (DataFrame, str, numpy.ndarray) -> None
        self.data_frame = data_frame
        self.column_name = column_name
        self.bins = bins
        series = self.data_frame[column_name]
        self.sorted_series = series.sort_values()  # type: Series


def create_report(report_data_list, report_config):
    # type: (List[ReportData], ReportConfig) -> None
    """Create a report from the coco_eval object, storing it in out_dir.

    The report will consist of a series of HTML pages, images, and maybe
    other files. If out_dir already  exists, this function will throw an
    exception.

    :param coco_eval: An evaluation object from which the report will be
    generated
    :param report_config: See ReportConfig; includes the path of the output
    directory that will be created
    :return: None
    """
    output_paths = OutputPaths(report_config.out_dir)
    data_frame = _create_main_data_frame(report_data_list)
    with open(output_paths.report_index_path, 'w') as report_index_file:
        _add_all_runs_table(report_index_file, data_frame, report_data_list)
        #_add_all_runs__metric_pages(report_index_file, output_paths,
        #                            report_config, data_frame)
        _add_all_run_pairs_metric_pages(
            report_index_file, output_paths, data_frame, report_data_list)
        _add_single_run_metric_pages(
            report_index_file, output_paths, report_config, data_frame,
            report_data_list)
        _add_unlabeled_images(report_index_file, output_paths.image_dir)
    _create_image_reports(output_paths, data_frame)


def _add_all_run_pairs_metric_pages(
        report_index_file, output_paths, data_frame, report_data_list):
    # type: (IO, OutputPaths, DataFrame, List[ReportData]) -> None
    _write_header(report_index_file, 'Comparison of pairs of runs')
    run_names = data_frame.columns.levels[0]
    if len(run_names) < 2:
        return
    # Only handle comparison of the first two methods for now.
    all_pair_indexes = [[0, 1]]
    for pair_indexes in all_pair_indexes:
        pair_names = run_names[pair_indexes]
        pair_name = '_VS_'.join(pair_names)
        pair_output_paths = RunOutputPaths(output_paths, pair_name)
        pair_report_data_list = [report_data_list[i] for i in pair_indexes]
        _add_run_pair_metric_page(
            report_index_file, pair_output_paths, pair_name,
            data_frame[pair_names], pair_report_data_list)


def _add_single_run_metric_pages(report_index_file, output_paths, report_config,
                                 data_frame, report_data_list):
    # type: (IO, OutputPaths, ReportConfig, DataFrame, List[ReportData]) -> None
    _write_header(report_index_file, 'Reports per run')
    run_names = data_frame.columns.levels[0]
    for run_name, report_data in zip(run_names, report_data_list):
        run_output_paths = RunOutputPaths(output_paths, run_name)
        _add_single_run_metric_page(
            report_index_file, run_output_paths, report_config, run_name,
            data_frame[run_name], report_data)


def _add_run_pair_metric_page(
        report_index_file, pair_output_paths, pair_name, pair_data_frame,
        pair_report_data_list):
    # type: (IO, RunOutputPaths, str, DataFrame, List[ReportData]) -> None
    pair_index_path = pair_output_paths.index_path
    out_dir = pair_output_paths.output_paths.out_dir
    report_index_file.write('<a href="%s">%s</a><br>\n' % (
        os.path.relpath(pair_index_path, out_dir), pair_name))
    with open(pair_index_path, 'w') as pair_index_file:
        _write_header(pair_index_file, pair_name)
        _add_run_pair_table(pair_index_file, pair_data_frame,
                            pair_report_data_list)
        _add_run_pair_metric_pages(pair_index_file, pair_output_paths,
                                   pair_data_frame)


def _add_run_pair_metric_pages(html_file, pair_output_paths, pair_data_frame):
    # type: (IO, RunOutputPaths, DataFrame) -> None
    _write_header(html_file, 'Distribution of metric differences (except '
                             'SPICE-related, which come later below)')
    run_names = pair_data_frame.columns.levels[0]
    for column_name in COLUMNS_FOR_HISTOGRAM_NON_SPICE:
        metric_data = _create_difference_metric_data(
            run_names, column_name, pair_data_frame)
        _add_metric_page(html_file, pair_output_paths, metric_data)
    _write_header(html_file, 'Distribution of SPICE-related metric '
                             'differences')
    for column_name in COLUMNS_FOR_HISTOGRAM_SPICE:
        metric_data = _create_difference_metric_data(
            run_names, column_name, pair_data_frame)
        _add_metric_page(html_file, pair_output_paths, metric_data)


def _create_difference_metric_data(run_names, column_name, pair_data_frame):
    # type: (List[str], str, DataFrame) -> MetricData
    first_run_name = run_names[0]
    second_run_name = run_names[1]
    difference_name = '%s_minus_%s' % (first_run_name, second_run_name)
    difference_column_name = '%s_%s' % (column_name, difference_name)
    difference_series = (
            pair_data_frame[first_run_name][column_name] -
            pair_data_frame[second_run_name][column_name])
    difference_data_frame = difference_series.to_frame(
        difference_column_name)  # type: DataFrame
    valid_difference_data_frame = difference_data_frame.dropna()
    valid_count = valid_difference_data_frame.count().item()
    n_bins = _n_bins_from_count(valid_count)
    _, bins = numpy.histogram(valid_difference_data_frame, bins=n_bins)
    return MetricData(difference_data_frame, difference_column_name, bins)


def _n_bins_from_count(count):
    # type: (int) -> int
    if count < 50:
        return 10
    if count < 500:
        return 20
    return 30


def _add_run_pair_table(html_file, pair_data_frame, pair_report_data_list):
    # type: (IO, DataFrame, List[ReportData]) -> None
    """Write overall measures, from COCOEvalCap.eval."""
    _write_header(html_file, 'Measures over all images')
    run_pair_summary = _create_run_pair_summary(
        pair_report_data_list, pair_data_frame)
    html_file.write(
        run_pair_summary.to_html(float_format=lambda x: '%.6f' % x) + '\n')


def _create_run_pair_summary(pair_report_data_list, pair_data_frame):
    # type(List[ReportData], str, DataFrame) -> DataFrame
    run_names = pair_data_frame.columns.levels[0]
    run_summaries = [
        _create_single_run_summary(
            report_data, run_name, pair_data_frame[run_name])
        for (run_name, report_data) in zip(run_names, pair_report_data_list)]
    mean_summaries = [
        pair_data_frame[run_name][ALL_SUMMARY_COLUMNS].mean().to_frame(
            run_name + '.means') for run_name in run_names]
    t_test_results = {column: ttest_rel(
        pair_data_frame[run_names[0]][column],
        pair_data_frame[run_names[1]][column], nan_policy='omit')
        for column in ALL_SUMMARY_COLUMNS}
    t_statistics = {
        column: statistic_and_p_value[0] for column, statistic_and_p_value in
        t_test_results.items()}
    p_values = {
        column: statistic_and_p_value[1] for column, statistic_and_p_value in
        t_test_results.items()}
    t_statistic_data_frame = Series(t_statistics).to_frame(
        't-test statistic (two-tailed)')
    p_value_data_frame = Series(p_values).to_frame('p-value (two-tailed)')
    return concat(run_summaries + mean_summaries + [
        t_statistic_data_frame, p_value_data_frame], axis=1)


def _add_single_run_metric_page(
        report_index_file, run_output_paths, report_config, run_name,
        single_run_data_frame, report_data):
    # type: (IO, RunOutputPaths, ReportConfig, str, DataFrame, ReportData) -> None
    run_index_path = run_output_paths.index_path
    out_dir = run_output_paths.output_paths.out_dir
    report_index_file.write('<a href="%s">%s</a><br>\n' % (
        os.path.relpath(run_index_path, out_dir), run_name))
    with open(run_index_path, 'w') as run_index_file:
        _write_header(run_index_file, run_name)
        _add_single_run_table(run_index_file, run_name,
                              single_run_data_frame, report_data)
        _add_metric_pages(run_index_file, run_output_paths, report_config,
                          single_run_data_frame)


def _add_single_run_table(
        html_file, run_name, single_run_data_frame, report_data):
    # type: (IO, str, DataFrame, ReportData) -> None
    """Write overall measures, from COCOEvalCap.eval."""
    _write_header(html_file, 'Measures over all images')
    single_run_summary = _create_single_run_summary(
            report_data, run_name, single_run_data_frame)
    html_file.write(single_run_summary.to_html() + '\n')


def _create_main_data_frame(report_data_list):
    # type: (List[ReportData]) -> DataFrame
    data_frames = [_create_run_data_frame(report_data) for report_data in
                   report_data_list]
    run_names = ['%s_%s' % (report_data.model_id, report_data.split) for
                 report_data in report_data_list]
    data_frame = concat(data_frames, keys=run_names, axis=1)
    return data_frame


def _create_run_data_frame(report_data):
    # type: (ReportData) -> DataFrame
    coco_eval = report_data.coco_eval
    data_frame = json_normalize(coco_eval.imgToEval.values())
    result = {image_id: annotation[0][RESULT_KEY_CAPTION] for
              (image_id, annotation) in coco_eval.cocoRes.imgToAnns.items()}
    ground_truth = {image_id: _ground_truth_captions(annotations)
                    for (image_id, annotations) in
                    coco_eval.coco.imgToAnns.items()}
    data_frame[EvalColumns.RESULT_CAPTION] = data_frame[
        EvalColumns.IMAGE_ID].map(result)
    data_frame[EvalColumns.GROUND_TRUTH_CAPTIONS] = data_frame[
        EvalColumns.IMAGE_ID].map(ground_truth)
    image_paths = {prediction[PREDICTION_KEY_IMAGE_ID]: os.path.join(
        report_data.image_root, prediction[PREDICTION_KEY_FILE_PATH])
        for prediction in report_data.predictions}
    data_frame[EvalColumns.PATH] = data_frame[EvalColumns.IMAGE_ID].map(
        image_paths)
    data_frame.set_index(INDEX_COLUMN_NAME, inplace=True)
    return data_frame


def _ground_truth_captions(annotations):
    # type (List[Dict]) -> List[str]
    return [annotation[GROUND_TRUTH_KEY_CAPTION] for annotation in annotations]


def _create_image_reports(output_paths, data_frame):
    # type: (OutputPaths, DataFrame) -> None
    for image_id in data_frame.index:
        image_data_frame = data_frame.loc[image_id].unstack(level=0)
        _create_image_report(output_paths, image_data_frame, image_id)


def _create_image_report(output_paths, image_data_frame, image_id):
    # type: (OutputPaths, DataFrame, int) -> None
    image_report_path = _image_report_path(
        output_paths.image_report_dir, str(image_id))
    with open(image_report_path, 'w') as image_report_file:
        image_dir_for_html = PathForHTML(output_paths.image_dir,
                                         output_paths.image_report_dir)
        run_names = image_data_frame.columns
        first_run_name = run_names[0]
        original_path = image_data_frame[first_run_name][EvalColumns.PATH]
        _copy_and_write_image(
            image_report_file, image_dir_for_html, original_path, image_id)
        _write_header(image_report_file, 'Generated caption(s)')
        for run_name in run_names:
            image_report_file.write('<b>%s</b>: ' % run_name)
            image_report_file.write(
                image_data_frame[run_name][EvalColumns.RESULT_CAPTION])
            image_report_file.write('<br>\n')
        _write_header(image_report_file, 'Ground truth captions')
        for ground_truth_caption in image_data_frame[first_run_name][
                EvalColumns.GROUND_TRUTH_CAPTIONS]:
            image_report_file.write(ground_truth_caption + '<br>\n')
        _write_header(image_report_file, 'Metrics')
        image_report_file.write(
            image_data_frame.to_html(float_format=lambda x: '%.6f' % x))


def _copy_and_write_image(image_report_file, image_dir_for_html,
                          original_path, image_id):
    # type: (IO, PathForHTML, str, int) -> None
    image_extension = os.path.splitext(original_path)[1]
    image_file_name = str(image_id) + image_extension
    image_path_for_html = image_dir_for_html.join(image_file_name)
    shutil.copyfile(original_path, image_path_for_html.regular)
    image_relative_path = image_path_for_html.relative()
    _write_html_image(image_report_file, image_relative_path,
                      image_relative_path, align=HTML_IMAGE_ALIGN_LEFT)


def _add_all_runs_table(html_file, data_frame, report_data_list):
    # type: (IO, DataFrame, List[ReportData]) -> None
    """Write overall measures, from COCOEvalCap.eval."""
    _write_header(html_file, 'Measures over all images')
    run_names = data_frame.columns.levels[0]
    summary_list = [
        _create_single_run_summary(
            report_data, run_name, data_frame[run_name])
        for (report_data, run_name) in zip(report_data_list, run_names)]
    summary_data_frame = concat(summary_list, axis=1)
    html_file.write(summary_data_frame.to_html() + '\n')


def _create_single_run_summary(report_data, run_name, single_run_data_frame):
    # type(ReportData, str, DataFrame) -> DataFrame
    extra_means = single_run_data_frame[EXTRA_SUMMARY_COLUMNS].mean()
    return Series(report_data.coco_eval.eval).rename(
        index={COCO_EVAL_SPICE_COLUMN: EvalColumns.SPICE}).append(
        extra_means).to_frame(run_name)


def _add_metric_pages(html_file, run_output_paths, report_config, data_frame):
    # type: (IO, RunOutputPaths, ReportConfig, DataFrame) -> None
    _write_header(html_file, 'Distribution of different measures (except '
                             'SPICE-related, which come later below)')
    for column_name in COLUMNS_FOR_HISTOGRAM_NON_SPICE:
        bins = report_config.histogram_bins[column_name]
        metric_data = MetricData(data_frame, column_name, bins)
        _add_metric_page(html_file, run_output_paths, metric_data)
    _write_header(html_file, 'Distribution of SPICE-related measures')
    for column_name in COLUMNS_FOR_HISTOGRAM_SPICE:
        bins = report_config.histogram_bins[column_name]
        metric_data = MetricData(data_frame, column_name, bins)
        _add_metric_page(html_file, run_output_paths, metric_data)


def _add_metric_page(html_file, run_output_paths, metric_data):
    # type: (IO, RunOutputPaths, MetricData) -> None
    figure = _plot_histogram(metric_data.data_frame, metric_data.bins,
                             metric_data.column_name)
    image_path = run_output_paths.histogram_image_path(metric_data.column_name)
    figure.savefig(image_path)
    # Close the figure so that matplotlib doesn't complain about memory.
    plt.close(figure)
    metric_path = run_output_paths.metric_path(metric_data.column_name)
    run_dir = run_output_paths.run_dir
    relative_image_path = os.path.relpath(image_path, run_dir)
    relative_metric_path = os.path.relpath(metric_path, run_dir)
    _write_html_image(html_file, relative_image_path, relative_metric_path)
    relative_image_report_dir = os.path.relpath(
        run_output_paths.output_paths.image_report_dir, run_dir)
    _create_metric_html(metric_path, relative_image_path,
                        relative_image_report_dir, metric_data)


def _create_metric_html(metric_path, relative_image_path,
                        relative_image_report_dir, metric_data):
    # type: (str, str, str, MetricData) -> None
    with open(metric_path, 'w') as metric_file:
        _write_html_image(
            metric_file, relative_image_path, relative_image_path,
            align=HTML_IMAGE_ALIGN_LEFT)
        _print_metric_stats(metric_file, metric_data.sorted_series)
        metric_file.write('\n<br>')
        _write_sorted_images(metric_file, relative_image_report_dir,
                             metric_data.sorted_series, metric_data.bins)
        _write_many_line_breaks(metric_file)


def _write_many_line_breaks(html_file):
    # type: (IO) -> None
    html_file.write('<br>\n' * 100)


def _write_sorted_images(
        metric_file, relative_image_report_dir, sorted_series, bins):
    # type: (IO, str, Series, numpy.ndarray) -> None
    extended_bins = numpy.concatenate(([numpy.NINF], bins, [numpy.Inf]))
    bins_start_end = zip(extended_bins[:-1], extended_bins[1:])
    bin_names = ['%.2f_to_%.2f' % (start, end) for (start, end) in
                 bins_start_end]
    _write_header(metric_file, 'Images per bucket')
    _write_sorted_images_anchor_links(metric_file, bin_names)
    _write_header(metric_file, 'Images')
    for ((start, end), name)in zip(bins_start_end, bin_names):
        _write_anchor(metric_file, name)
        _write_header(metric_file, name)
        filtered_series = sorted_series[
            lambda metric: (metric >= start) & (metric < end)]
        _write_image_series(metric_file, relative_image_report_dir,
                            filtered_series)


def _write_image_series(metric_file, relative_image_report_dir, series):
    # type: (IO, str, Series) -> None
    for image_id, metric in series.iteritems():
        relative_image_report_path = _image_report_path(
            relative_image_report_dir, image_id)
        metric_file.write('<a href="%s">%s</a>: %f, \n' % (
            relative_image_report_path, image_id, metric))


def _image_report_path(relative_image_report_dir, image_id):
    # type: (str, str) -> str
    return os.path.join(relative_image_report_dir, '%s.html' % image_id)


def _write_anchor(html_file, name):
    # type: (IO, str) -> None
    html_file.write('<a name="%s"></a>' % name)


def _write_sorted_images_anchor_links(metric_file, bin_names):
    # type: (IO, List[str]) -> None
    anchor_links = [_create_anchor_link(bin_name) for bin_name in bin_names]
    metric_file.write(' || '.join(anchor_links))


def _create_anchor_link(bin_name):
    # type: (str) -> str
    return '<a href="#%s">%s</a>\n' % (bin_name, bin_name)


def _print_metric_stats(metric_file, series):
    # type: (IO, Series) -> None
    stats = series.describe()
    _write_header(metric_file, 'Metric stats')
    metric_file.write(stats.to_frame().to_html(
        float_format=lambda x: '%.6f' % x) + '\n')


def _plot_histogram(data_frame, bins, column_name):
    # type: (DataFrame, Optional[numpy.ndarray], str) -> Figure
    figure = plt.figure()
    # Draw the histogram into the current active axes.
    data_frame.hist(column=column_name, ax=plt.gca(), edgecolor='black',
                    linewidth=1.2, grid=False, bins=bins)
    count = data_frame[column_name].count()
    plt.title('%s (image count = %d)' % (column_name, count))
    plt.ylabel('Frequency')
    plt.xlabel(column_name)
    return figure


def _write_html_image(html_file, image_src, link, align=None):
    # type: (IO, str, str, Optional[str]) -> None
    if align:
        align_string = "align='%s'" % align
    else:
        align_string = ''
    html_file.write("<a href='%s'><img src='%s' width='%d' %s></a>\n" % (
        link, image_src, HTML_IMAGE_WIDTH_PIXELS, align_string))


def _add_unlabeled_images(html_file, image_dir):
    # type: (IO, str) -> None
    _write_header(html_file, 'Unlabeled images')
    html_file.write('to be added at some point?')
    # TODO: implement this to allow us to view unlabeled images (no ground
    # truth).


def _write_header(html_file, header):
    # type: (IO, str) -> None
    html_file.write('<h1>%s</h1>\n' % header)
