import sys
import os
from pandas.io.json import json_normalize
sys.path.append("coco-caption")
from pycocoevalcap.eval import COCOEvalCap

INDEX_HTML = 'index.html'
IMAGE_ID_COLUMN_NAME = 'image_id'
INDEX_COLUMN_NAME = IMAGE_ID_COLUMN_NAME


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
    data_frame = json_normalize(coco_eval.imgToEval.values())
    data_frame.set_index(INDEX_COLUMN_NAME, inplace=True)
    report_index_path = os.path.join(out_dir, INDEX_HTML)
    with open(report_index_path, 'w') as report_index_file:
        report_index_file.write('This is a table with the data that was read')
        report_index_file.write(data_frame.to_html())
