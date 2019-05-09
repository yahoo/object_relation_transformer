import numpy
import os
import sys
import matplotlib.pyplot as plt
# By using Agg, pyplot will not try to open a display
plt.switch_backend('Agg')
from matplotlib.figure import Figure
from pandas import DataFrame, Series
from pandas.io.json import json_normalize
from typing import IO, Optional, Dict
sys.path.append("coco-caption")
from pycocoevalcap.eval import COCOEvalCap

INDEX_HTML = 'index.html'
HTML_IMAGE_WIDTH_PIXELS = 400
SUMMARY_VALUES_COLUMN_NAME = 'Value'
PLOT_DIR_NAME = 'plots'


class EvalColumns:
    IMAGE_ID = 'image_id'
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
        EvalColumns.CIDER: numpy.arange(0, 6, 0.25),
        EvalColumns.BLEU1: numpy.arange(0, 1, 0.05),
        EvalColumns.BLEU2: numpy.arange(0, 1, 0.05),
        EvalColumns.BLEU3: numpy.arange(0, 1, 0.05),
        EvalColumns.BLEU4: numpy.arange(0, 1, 0.05),
        EvalColumns.METEOR: numpy.arange(0, 1, 0.05),
        EvalColumns.ROUGE_L: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_PR: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_RE: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_OBJECT: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_OBJECT_PR: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_OBJECT_RE: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_ATTRIBUTE: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_ATTRIBUTE_PR: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_ATTRIBUTE_RE: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_RELATION: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_RELATION_PR: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_RELATION_RE: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_SIZE: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_SIZE_PR: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_SIZE_RE: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_CARDINALITY: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_CARDINALITY_PR: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_CARDINALITY_RE: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_COLOR: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_COLOR_PR: numpy.arange(0, 1, 0.05),
        EvalColumns.SPICE_COLOR_RE: numpy.arange(0, 1, 0.05)
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
    os.makedirs(out_dir)
    report_index_path = os.path.join(out_dir, INDEX_HTML)
    with open(report_index_path, 'w') as report_index_file:
        _add_summary_table(report_index_file, coco_eval)
        data_frame = json_normalize(coco_eval.imgToEval.values())
        data_frame.set_index(INDEX_COLUMN_NAME, inplace=True)
        plot_dir_for_html = PathForHTML.from_base_and_relative(
            out_dir, PLOT_DIR_NAME)
        os.makedirs(plot_dir_for_html.regular)
        _add_histograms(report_index_file, plot_dir_for_html,
                        report_config, data_frame)
        _add_unlabeled_images(report_index_file, out_dir)


def _add_summary_table(html_file, coco_eval):
    # type: (IO, COCOEvalCap) -> None
    """Write overall measures, from COCOEvalCap.eval."""
    _write_header(html_file, 'Measures over all images')
    summary_data_frame = Series(coco_eval.eval).to_frame(
        SUMMARY_VALUES_COLUMN_NAME)
    html_file.write(summary_data_frame.to_html() + '\n')


def _add_histograms(html_file, plot_dir_for_html, report_config, data_frame):
    # type: (IO, PathForHTML, ReportConfig, DataFrame) -> None
    _write_header(html_file, 'Distribution of different measures (except '
                             'SPICE-related, which come later below)')
    for column_name in COLUMNS_FOR_HISTOGRAM_NON_SPICE:
        bins = report_config.histogram_bins.get(column_name)
        _add_histogram(html_file, plot_dir_for_html,
                       bins, data_frame, column_name)
    _write_header(html_file, 'Distribution of SPICE-related measures')
    for column_name in COLUMNS_FOR_HISTOGRAM_SPICE:
        _add_histogram(html_file, plot_dir_for_html,
                       report_config.histogram_bins.get(column_name),
                       data_frame, column_name)


def _add_histogram(html_file, plot_dir_for_html, bins, data_frame, column_name):
    # type: (IO, PathForHTML, Optional[numpy.ndarray], DataFrame, str) -> None
    figure = _plot_histogram(data_frame, bins, column_name)
    image_file_name = column_name + '.jpg'
    image_path_for_html = plot_dir_for_html.join(image_file_name)
    _save_figure(html_file, image_path_for_html, figure)


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


def _save_figure(html_file, image_path_for_html, figure):
    # type: (IO, PathForHTML, Figure) -> None
    figure.savefig(image_path_for_html.regular)
    html_file.write("<img src='%s' width='%d'>\n" % (
        image_path_for_html.relative, HTML_IMAGE_WIDTH_PIXELS))


def _add_unlabeled_images(html_file, out_dir):
    # type: (IO, str) -> None
    _write_header(html_file, 'Unlabeled images')
    html_file.write('to be added at some point?')
    # TODO: implement this to allow us to view unlabeled images (no ground
    # truth).


def _write_header(html_file, header):
    # type: (IO, str) -> None
    html_file.write('<h1>%s</h1>\n' % header)
