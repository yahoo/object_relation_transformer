import sys
sys.path.append("coco-caption")
from pycocoevalcap.eval import COCOEvalCap


class CorrectCOCOEvalCap(COCOEvalCap):
    """This class inherits from COCOEvalCap in order to fix an issue
    with that class's implementation, without having to go into the
    coco-caption/ codebase. The COCOEvalCap implementation has a problem in
    how it assigns SPICE scores to the images, so that occasionally the
    computed SPICE scores end up getting assigned to the wrong images."""

    # This has to match the string for SPICE in COCOEvalCap, which is
    # hard-coded there.
    METHOD_SPICE = 'SPICE'

    # OVERRIDES: this function overrides setImgToEvalImgs() from COCOEvalCap,
    # in order to fix the ordering of the SPICE scores.
    def setImgToEvalImgs(self, scores, imgIds, method):
        if method == CorrectCOCOEvalCap.METHOD_SPICE:
            # The SPICE scores are actually ordered according the the sorted
            # imgIds, and not according the the original imgIds order.
            imgIds = sorted(imgIds)
        COCOEvalCap.setImgToEvalImgs(self, scores, imgIds, method)
