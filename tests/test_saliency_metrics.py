import torch

from src.saliency_metrics import pointing_game_hit, bbox_iou_at_threshold


def test_pointing_game_hit_inside_box():
    heat = torch.zeros(224, 224)
    heat[100, 100] = 1.0
    bbox_mask = torch.zeros(224, 224, dtype=torch.bool)
    bbox_mask[80:120, 80:120] = True
    assert pointing_game_hit(heat, bbox_mask) is True


def test_pointing_game_hit_outside_box():
    heat = torch.zeros(224, 224)
    heat[10, 10] = 1.0
    bbox_mask = torch.zeros(224, 224, dtype=torch.bool)
    bbox_mask[100:150, 100:150] = True
    assert pointing_game_hit(heat, bbox_mask) is False


def test_bbox_iou_full_overlap():
    # Hot region is 40x40=1600 px. Use top_fraction so that k≈1600 (all hot
    # pixels selected, none of the zero background), giving IoU≈1.0.
    heat = torch.zeros(224, 224)
    heat[80:120, 80:120] = 1.0
    bbox_mask = torch.zeros(224, 224, dtype=torch.bool)
    bbox_mask[80:120, 80:120] = True
    iou = bbox_iou_at_threshold(heat, bbox_mask, top_fraction=0.032)
    assert iou > 0.99


def test_bbox_iou_no_overlap():
    heat = torch.zeros(224, 224)
    heat[:50, :50] = 1.0
    bbox_mask = torch.zeros(224, 224, dtype=torch.bool)
    bbox_mask[150:200, 150:200] = True
    iou = bbox_iou_at_threshold(heat, bbox_mask, top_fraction=0.05)
    assert iou == 0.0
