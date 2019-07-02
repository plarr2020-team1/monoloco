

from utils.iou import get_iou_matrix
from utils.camera import pixel_to_camera


def test_iou():
    boxes_pred = [[1, 100, 1, 200]]
    boxes_gt = [[100., 120., 150., 160.],[12, 110, 130., 160.]]
    iou_matrix = get_iou_matrix(boxes_pred, boxes_gt)
    assert iou_matrix.shape == (len(boxes_pred), len(boxes_gt))


def test_pixel_to_camera():
    kk = [[718.3351, 0., 600.3891], [0., 718.3351, 181.5122], [0., 0., 1.]]
    zz = 10
    uv_vector = [1000., 400.]
    xx_norm = pixel_to_camera(uv_vector, kk, 1)[0]
    xx_1 = xx_norm * zz
    xx_2 = pixel_to_camera(uv_vector, kk, zz)[0]
    assert xx_1 == xx_2
