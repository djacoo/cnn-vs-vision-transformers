"""Loader for Oxford-IIIT Pet head bounding-box annotations (Pascal-VOC XML)."""
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_voc_xml(path):
    """Return a dict with width, height, xmin, ymin, xmax, ymax for one annotation."""
    root = ET.parse(path).getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    obj = root.find("object")
    bb = obj.find("bndbox")
    return {
        "width": width,
        "height": height,
        "xmin": int(bb.findtext("xmin")),
        "ymin": int(bb.findtext("ymin")),
        "xmax": int(bb.findtext("xmax")),
        "ymax": int(bb.findtext("ymax")),
    }


def load_all_bboxes(data_root="data"):
    """Return {image_stem: bbox_dict} for every XML under data/oxford-iiit-pet/annotations/xmls."""
    xml_dir = Path(data_root) / "oxford-iiit-pet" / "annotations" / "xmls"
    return {p.stem: parse_voc_xml(str(p)) for p in xml_dir.glob("*.xml")}


def bbox_to_resized_mask(bbox, target_size=224):
    """Return a (target_size, target_size) bool tensor representing the bbox.

    Mimics the transform pipeline: Resize(256) + CenterCrop(target_size).
    """
    import torch

    w0, h0 = bbox["width"], bbox["height"]
    short = min(w0, h0)
    scale = 256 / short
    w1, h1 = int(round(w0 * scale)), int(round(h0 * scale))

    x0 = bbox["xmin"] * scale
    y0 = bbox["ymin"] * scale
    x1 = bbox["xmax"] * scale
    y1 = bbox["ymax"] * scale

    off_x = (w1 - target_size) / 2
    off_y = (h1 - target_size) / 2
    x0, x1 = max(0, x0 - off_x), min(target_size, x1 - off_x)
    y0, y1 = max(0, y0 - off_y), min(target_size, y1 - off_y)

    mask = torch.zeros(target_size, target_size, dtype=torch.bool)
    if x1 > x0 and y1 > y0:
        mask[int(y0):int(y1), int(x0):int(x1)] = True
    return mask
