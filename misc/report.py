import sys
import os
import matplotlib.pyplot as plt
# By using Agg, pyplot will not try to open a display
plt.switch_backend('Agg')
from matplotlib.figure import Figure
from pandas import DataFrame, Series
from pandas.io.json import json_normalize
from typing import IO
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
    SPICE_OBJECT = 'SPICE.Object.f'
    SPICE_RELATION = 'SPICE.Relation.f'
    SPICE_ATTRIBUTE = 'SPICE.Attribute.f'
    SPICE_COLOR = 'SPICE.Color.f'
    SPICE_CARDINALITY = 'SPICE.Cardinality.f'
    SPICE_SIZE = 'SPICE.Size.f'


INDEX_COLUMN_NAME = EvalColumns.IMAGE_ID
COLUMNS_FOR_HISTOGRAM_NON_SPICE = [
    EvalColumns.BLEU1, EvalColumns.BLEU2, EvalColumns.BLEU3, EvalColumns.BLEU4,
    EvalColumns.CIDER, EvalColumns.METEOR, EvalColumns.ROUGE_L]
COLUMNS_FOR_HISTOGRAM_SPICE = [
    EvalColumns.SPICE, EvalColumns.SPICE_OBJECT, EvalColumns.SPICE_RELATION,
    EvalColumns.SPICE_ATTRIBUTE, EvalColumns.SPICE_COLOR,
    EvalColumns.SPICE_CARDINALITY, EvalColumns.SPICE_SIZE]


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


def create_report(coco_eval, out_dir):
    # type: (COCOEvalCap, str) -> None
    """Create a report from the coco_eval object, storing it in out_dir.

    The report will consist of a series of HTML pages, images, and maybe
    other files. If out_dir already  exists, this function will throw an
    exception.

    :param coco_eval: An evaluation object from which the report will be
    generated
    :param out_dir: the path of the output directory that will be created
    :return: None
    """
    # For now, just fail if the directory already exists, so we don't
    # overwrite anything accidentally.
    os.makedirs(out_dir)
    report_index_path = os.path.join(out_dir, INDEX_HTML)
    with open(report_index_path, 'w') as report_index_file:
        _add_summary_table(report_index_file, coco_eval)
        data_frame = json_normalize(coco_eval.imgToEval.values())
        data_frame.set_index(INDEX_COLUMN_NAME, inplace=True)
        plot_dir_for_html = PathForHTML.from_base_and_relative(
            out_dir, PLOT_DIR_NAME)
        os.makedirs(plot_dir_for_html.regular)
        _add_histograms(report_index_file, plot_dir_for_html, data_frame)
        _add_unlabeled_images(report_index_file, out_dir)


def _add_summary_table(html_file, coco_eval):
    # type: (IO, COCOEvalCap) -> None
    """Write overall measures, from COCOEvalCap.eval."""
    _write_header(html_file, 'Measures over all images')
    summary_data_frame = Series(coco_eval.eval).to_frame(
        SUMMARY_VALUES_COLUMN_NAME)
    html_file.write(summary_data_frame.to_html() + '\n')


def _add_histograms(html_file, plot_dir_for_html, data_frame):
    # type: (IO, PathForHTML, DataFrame) -> None
    _write_header(html_file, 'Distribution of different measures (except '
                             'SPICE-related, which come later below)')
    for column_name in COLUMNS_FOR_HISTOGRAM_NON_SPICE:
        _add_histogram(html_file, data_frame, column_name,
                       plot_dir_for_html)
    _write_header(html_file, 'Distribution of SPICE-related measures')
    for column_name in COLUMNS_FOR_HISTOGRAM_SPICE:
        _add_histogram(html_file, data_frame, column_name,
                       plot_dir_for_html)


def _add_histogram(html_file, data_frame, column_name, plot_dir_for_html):
    # type: (IO, DataFrame, str, PathForHTML) -> None
    figure = _plot_histogram(data_frame, column_name)
    image_file_name = column_name + '.jpg'
    image_path_for_html = plot_dir_for_html.join(image_file_name)
    _save_figure(html_file, image_path_for_html, figure)


def _plot_histogram(data_frame, column_name):
    # type: (DataFrame, str) -> Figure
    figure = plt.figure()
    # Draw the histogram into the current active axes.
    data_frame.hist(column=column_name, ax=plt.gca(), edgecolor='black',
                    linewidth=1.2, grid=False)
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
