import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Union
from torchvision.ops import nms
import os


THIS_DIR = os.path.dirname(__file__)
WEIGHTS = os.path.join(THIS_DIR, "epoch3-3.pt")
model = YOLO(WEIGHTS)
CONF_THRESHOLD = 0.07
IOU_THRESHOLD = 0.05


def infer_image_bbox(image: np.ndarray) -> List[dict]:
    """Инференс YOLOv8 на изображении любого размера с нарезкой на тайлы 640x640 и NMS."""
    res_list = []

    h_img, w_img = image.shape[:2]
    tile_size = 640
    stride = tile_size

    tiles = []
    tile_coords = []
    for y in range(0, h_img, stride):
        for x in range(0, w_img, stride):
            tile = image[y:min(y + tile_size, h_img),
                         x:min(x + tile_size, w_img)]
            tiles.append(tile)
            tile_coords.append((x, y))

    results = model.predict(
        source=tiles,
        imgsz=tile_size,
        conf=CONF_THRESHOLD,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )

    raw_boxes = []
    raw_scores = []

    for (res, (x_off, y_off)) in zip(results, tile_coords):
        for box in res.boxes:
            label = int(box.cls[0])
            if label != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            x1_global = x1 + x_off
            x2_global = x2 + x_off
            y1_global = y1 + y_off
            y2_global = y2 + y_off

            raw_boxes.append([x1_global, y1_global, x2_global, y2_global])
            raw_scores.append(conf)

    if raw_boxes:
        boxes_tensor = torch.tensor(raw_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)

        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=IOU_THRESHOLD)

        for idx in keep_indices:
            x1_global, y1_global, x2_global, y2_global = boxes_tensor[idx].tolist()
            conf = scores_tensor[idx].item()

            xc = ((x1_global + x2_global) / 2) / w_img
            yc = ((y1_global + y2_global) / 2) / h_img
            w = (x2_global - x1_global) / w_img
            h = (y2_global - y1_global) / h_img

            res_list.append({
                'label': 0,
                'xc': xc,
                'yc': yc,
                'w': w,
                'h': h,
                'score': conf,
                'w_img': w_img,
                'h_img': h_img
            })

    if not res_list:
        res_list.append({
            'label': 0,
            'xc': None,
            'yc': None,
            'w': None,
            'h': None,
            'score': None,
            'w_img': w_img,
            'h_img': h_img
        })

    return res_list

######## НИЖЕ ЧАСТЬ С ПЕРЕКРЫТИЕМ ##############

# def infer_image_bbox(image: np.ndarray) -> List[dict]:
#     """Инференс YOLOv8 на изображении любого размера с нарезкой на тайлы 640x640,
#        перекрытием и NMS для объединения дублей.
#     """
#     res_list = []

#     h_img, w_img = image.shape[:2]
#     tile_size = 640
#     stride = 480  # перекрытие 160 пикселей

#     tiles = []
#     tile_coords = []  # (x_offset, y_offset)

#     # Разрезка на тайлы с перекрытием
#     for y in range(0, h_img, stride):
#         for x in range(0, w_img, stride):
#             tile = image[y:min(y + tile_size, h_img),
#                          x:min(x + tile_size, w_img)]
#             tiles.append(tile)
#             tile_coords.append((x, y))

#     # Прогон всех тайлов через YOLO
#     results = model.predict(
#         source=tiles,
#         imgsz=tile_size,
#         conf=CONF_THRESHOLD,
#         device=0 if torch.cuda.is_available() else "cpu",
#         verbose=False
#     )

#     raw_boxes = []
#     raw_scores = []

#     # Сбор всех боксов в глобальных координатах
#     for (res, (x_off, y_off)) in zip(results, tile_coords):
#         for box in res.boxes:
#             label = int(box.cls[0])
#             if label != 0:
#                 continue

#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             conf = float(box.conf[0])

#             x1_global = x1 + x_off
#             x2_global = x2 + x_off
#             y1_global = y1 + y_off
#             y2_global = y2 + y_off

#             raw_boxes.append([x1_global, y1_global, x2_global, y2_global])
#             raw_scores.append(conf)

#     if raw_boxes:
#         boxes_tensor = torch.tensor(raw_boxes, dtype=torch.float32)
#         scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)

#         keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.1)

#         for idx in keep_indices:
#             x1_global, y1_global, x2_global, y2_global = boxes_tensor[idx].tolist()
#             conf = scores_tensor[idx].item()

#             xc = ((x1_global + x2_global) / 2) / w_img
#             yc = ((y1_global + y2_global) / 2) / h_img
#             w = (x2_global - x1_global) / w_img
#             h = (y2_global - y1_global) / h_img

#             res_list.append({
#                 'label': 0,
#                 'xc': xc,
#                 'yc': yc,
#                 'w': w,
#                 'h': h,
#                 'score': conf,
#                 'w_img': w_img,
#                 'h_img': h_img
#             })

#     # Если нет детекций
#     if not res_list:
#         res_list.append({
#             'label': 0,
#             'xc': None,
#             'yc': None,
#             'w': None,
#             'h': None,
#             'score': None,
#             'w_img': w_img,
#             'h_img': h_img
#         })

#     return res_list



def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    """Функция производит инференс модели на одном или нескольких изображениях.

    Args:
        images (Union[List[np.ndarray], np.ndarray]): Список изображений или одно изображение.

    Returns:
        List[List[dict]]: Список списков словарей с результатами предикта 
        на найденных изображениях.
        Пример выходных данных:
        [
            [
                {
                    'xc': 0.5,
                    'yc': 0.5,
                    'w': 0.2,
                    'h': 0.3,
                    'label': 0,
                    'score': 0.95
                },
                ...
            ],
            ...
        ]
    """    
    results = []
    if isinstance(images, np.ndarray):
        images = [images]

    for image in images:        
        image_results = infer_image_bbox(image)
        results.append(image_results)
    
    return results