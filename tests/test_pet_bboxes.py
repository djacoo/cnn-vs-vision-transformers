import pytest

from src.pet_bboxes import parse_voc_xml


def test_parse_voc_xml(tmp_path):
    xml = """<annotation>
      <filename>Abyssinian_1.jpg</filename>
      <size><width>500</width><height>375</height></size>
      <object><name>cat</name>
        <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>120</ymax></bndbox>
      </object>
    </annotation>"""
    path = tmp_path / "Abyssinian_1.xml"
    path.write_text(xml)
    box = parse_voc_xml(str(path))
    assert box["width"] == 500
    assert box["height"] == 375
    assert box["xmin"] == 10
    assert box["xmax"] == 110


@pytest.mark.slow
def test_load_all_bboxes_reads_pet_xmls():
    from src.pet_bboxes import load_all_bboxes
    boxes = load_all_bboxes("data")
    assert len(boxes) > 3000
    sample = next(iter(boxes.values()))
    assert {"xmin", "ymin", "xmax", "ymax", "width", "height"} <= sample.keys()


def test_bbox_to_resized_mask_centered():
    from src.pet_bboxes import bbox_to_resized_mask
    box = {"width": 500, "height": 500, "xmin": 200, "ymin": 200,
           "xmax": 300, "ymax": 300}
    mask = bbox_to_resized_mask(box, target_size=224)
    assert mask.any()
    rows = mask.any(1).nonzero().flatten()
    assert abs(rows.min().item() - (224 - rows.numel()) // 2) <= 2
