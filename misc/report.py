import numpy
import os
import sys
import matplotlib.pyplot as plt
# By using Agg, pyplot will not try to open a display
plt.switch_backend('Agg')
from matplotlib.figure import Figure
from pandas import DataFrame, Series
from pandas.io.json import json_normalize
from typing import IO, Optional, Dict, List
sys.path.append("coco-caption")
from pycocoevalcap.eval import COCOEvalCap

INDEX_HTML = 'index.html'
HTML_IMAGE_WIDTH_PIXELS = 400
SUMMARY_VALUES_COLUMN_NAME = 'Value'
PLOT_DIR_NAME = 'plots'
IMAGE_REPORT_DIR_NAME = 'image_reports'
RESULT_KEY_CAPTION = 'caption'
GROUND_TRUTH_KEY_CAPTION = 'caption'


class EvalColumns:
    IMAGE_ID = 'image_id'
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


class PathForHTML:
    """Keeps track of a regular path, as well as a relative path, which can be
    used to create links in HTML code."""

    def __init__(self, regular, relative):
        # type: (str, str) -> None
        self.regular = regular
        self.relative = relative

    @staticmethod
    def from_base_and_relative(base, relative):
        # type: (str, str) -> PathForHTML
        return PathForHTML(os.path.join(base, relative), relative)

    def join(self, to_append):
        # type: (str) -> PathForHTML
        return PathForHTML(os.path.join(self.regular, to_append),
                           os.path.join(self.relative, to_append))


def create_report(coco_eval, report_config):
    # type: (COCOEvalCap, ReportConfig) -> None
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
    # For now, just fail if the directory already exists, so we don't
    # overwrite anything accidentally.
    out_dir = report_config.out_dir
    # output_directories = OutputDirectories(out_dir)
    os.makedirs(out_dir)
    out_dir_for_html = PathForHTML(out_dir, '.')
    image_report_dir_for_html = out_dir_for_html.join(IMAGE_REPORT_DIR_NAME)
    os.makedirs(image_report_dir_for_html.regular)
    report_index_path = os.path.join(out_dir, INDEX_HTML)
    data_frame = _create_main_data_frame(coco_eval)
    _create_image_reports(image_report_dir_for_html, data_frame)
    with open(report_index_path, 'w') as report_index_file:
        _add_summary_table(report_index_file, coco_eval)
        _add_metric_pages(report_index_file, out_dir_for_html,
                          image_report_dir_for_html, report_config,
                          data_frame)
        _add_unlabeled_images(report_index_file, out_dir)


def _create_main_data_frame(coco_eval):
    # type: (COCOEvalCap) -> DataFrame
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
    data_frame.set_index(INDEX_COLUMN_NAME, inplace=True)
    return data_frame


def _ground_truth_captions(annotations):
    # type (List[Dict]) -> List[str]
    return [annotation[GROUND_TRUTH_KEY_CAPTION] for annotation in annotations]


def _create_image_reports(image_report_dir_for_html, data_frame):
    # type: (PathForHTML, DataFrame) -> None
    for image_id in data_frame.index:
        image_series = data_frame.loc[image_id]
        _create_image_report(image_report_dir_for_html, image_series)


def _create_image_report(image_report_dir_for_html, image_series):
    # type: (PathForHTML, Series) -> None
    image_report_path = _image_report_path(
        image_report_dir_for_html, str(image_series.name))
    with open(image_report_path.regular, 'w') as image_report_file:
        _write_header(image_report_file, 'Generated caption')
        image_report_file.write(image_series[EvalColumns.RESULT_CAPTION])
        _write_header(image_report_file, 'Ground truth captions')
        for ground_truth_caption in image_series[
            EvalColumns.GROUND_TRUTH_CAPTIONS]:
            image_report_file.write(ground_truth_caption + '<br>\n')
        _write_header(image_report_file, 'Metrics')
        image_report_file.write(
            image_series.to_frame().to_html(float_format=lambda x: '%.6f' % x))


def _add_summary_table(html_file, coco_eval):
    # type: (IO, COCOEvalCap) -> None
    """Write overall measures, from COCOEvalCap.eval."""
    _write_header(html_file, 'Measures over all images')
    summary_data_frame = Series(coco_eval.eval).to_frame(
        SUMMARY_VALUES_COLUMN_NAME)
    html_file.write(summary_data_frame.to_html() + '\n')


def _add_metric_pages(html_file, out_dir_for_html, image_report_dir_for_html,
                      report_config, data_frame):
    # type: (IO, PathForHTML, PathForHTML, ReportConfig, DataFrame) -> None
    plot_dir_for_html = out_dir_for_html.join(PLOT_DIR_NAME)
    os.makedirs(plot_dir_for_html.regular)
    _write_header(html_file, 'Distribution of different measures (except '
                             'SPICE-related, which come later below)')
    for column_name in COLUMNS_FOR_HISTOGRAM_NON_SPICE:
        bins = report_config.histogram_bins[column_name]
        # metric_info = MetricInfo(data_frame, column_name, bins)
        _add_metric_page(html_file, out_dir_for_html, plot_dir_for_html,
                         image_report_dir_for_html, bins, data_frame,
                         column_name)
    _write_header(html_file, 'Distribution of SPICE-related measures')
    for column_name in COLUMNS_FOR_HISTOGRAM_SPICE:
        bins = report_config.histogram_bins[column_name]
        _add_metric_page(html_file, out_dir_for_html, plot_dir_for_html,
                         image_report_dir_for_html, bins, data_frame,
                         column_name)


def _add_metric_page(html_file, out_dir_for_html, plot_dir_for_html,
                     image_report_dir_for_html, bins, data_frame, column_name):
    # type: (IO, PathForHTML, PathForHTML, PathForHTML, numpy.ndarray, DataFrame, str) -> None
    figure = _plot_histogram(data_frame, bins, column_name)
    image_file_name = column_name + '.jpg'
    image_path_for_html = plot_dir_for_html.join(image_file_name)
    figure.savefig(image_path_for_html.regular)
    plt.close(figure)
    metric_file_name = column_name + '.html'
    metric_path_for_html = out_dir_for_html.join(metric_file_name)
    _save_figure(html_file, image_path_for_html.relative,
                 metric_path_for_html.relative)
    _create_metric_html(metric_path_for_html, image_path_for_html,
                        image_report_dir_for_html, data_frame[column_name],
                        bins)


def _create_metric_html(metric_path_for_html, image_path_for_html,
                        image_report_dir_for_html, series, bins):
    # type: (PathForHTML, PathForHTML, PathForHTML, Series, numpy.ndarray) -> None
    with open(metric_path_for_html.regular, 'w') as metric_file:
        _save_figure(metric_file, image_path_for_html.relative,
                     image_path_for_html.relative)
        sorted_series = series.sort_values()
        _print_metric_stats(metric_file, sorted_series)
        _write_sorted_images(metric_file, image_report_dir_for_html,
                             sorted_series, bins)
        _write_many_line_breaks(metric_file)


def _write_many_line_breaks(html_file):
    # type: (IO) -> None
    html_file.write('<br>\n' * 100)


def _write_sorted_images(metric_file, image_report_dir_for_html, sorted_series,
                         bins):
    # type: (IO, PathForHTML, Series, numpy.ndarray) -> None
    bins_start_end = zip(bins[:-1], bins[1:])
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
        _write_image_series(metric_file, image_report_dir_for_html,
                            filtered_series)


def _write_image_series(metric_file, image_report_dir_for_html, series):
    # type: (IO, PathForHTML, Series) -> None
    for image_id, metric in series.iteritems():
        image_report_path_for_html = _image_report_path(
            image_report_dir_for_html, image_id)
        metric_file.write('<a href="%s">%s</a>: %f, \n' % (
            image_report_path_for_html.relative, image_id, metric))


def _image_report_path(image_report_dir_for_html, image_id):
    # type: (PathForHTML, str) -> PathForHTML
    return image_report_dir_for_html.join('%s.html' % image_id)


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


def _save_figure(html_file, image_src, link):
    # type: (IO, str, str) -> None
    html_file.write("<a href='%s'><img src='%s' width='%d'></a>\n" % (
        link, image_src, HTML_IMAGE_WIDTH_PIXELS))


def _add_unlabeled_images(html_file, out_dir):
    # type: (IO, str) -> None
    _write_header(html_file, 'Unlabeled images')
    html_file.write('to be added at some point?')
    # TODO: implement this to allow us to view unlabeled images (no ground
    # truth).


def _write_header(html_file, header):
    # type: (IO, str) -> None
    html_file.write('<h1>%s</h1>\n' % header)
