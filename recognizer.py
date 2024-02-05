import cv2
import numpy as np
import easyocr
import torch
import os

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

def detect(img_in):
    path = os.path.dirname(__file__)
    with torch.no_grad():
        weights, imgsz = path + '/yolov5/weights/best.pt', 640
    
        device = select_device('cpu')
        conf_thres = 0.55
        iou_thres = 0.55
        classes = None
        agnostic_nms = False
        augment = False

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        # resizing image
        img = torch.zeros((3, imgsz, imgsz), device=device)  # init img
        img = letterbox(img_in, new_shape=imgsz)[0]

        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)  
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        # 0 - 255 to 0.0 - 1.0
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Inference
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        det = pred[0]
        # scaling coordinates back to the original image size
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_in.shape).round()
        det_boxes = det.cpu().numpy()

        # Extracting text using EasyOCR for each bounding box
        reader = easyocr.Reader(['en'], gpu=False)  # Specify the language(s) for OCR and disable GPU

        # Lists to store bounding box coordinates and recognized text values
    
        det_texts = []

        for i in range(len(det_boxes)):
            bbox = det[i, :4].detach().cpu().numpy().astype(int)


            x, y, w, h = bbox

            # Define a region around the bounding box
            region_margin = 100  # Adjust this margin based on your specific use case
            region_x = max(0, x - region_margin)
            region_y = max(0, y - region_margin)
            region_w = min(img_in.shape[1], w + region_margin)
            region_h = min(img_in.shape[0], h + region_margin)

            component_image = img_in[region_y:region_h, region_x:region_w]

            # Convert BGR to RGB (if needed)
            component_image_rgb = cv2.cvtColor(component_image, cv2.COLOR_BGR2RGB)

            # Perform OCR using EasyOCR
            result = reader.readtext(component_image_rgb)

            # Extract the text value if it exists
            text_value = result[0][1] if result else '??'

            # Adding bounding box coordinates to det_boxes array
            

            # Adding text value to det_texts array
            det_texts.append(text_value)

        # Convert the lists to NumPy arrays
        
        det_texts = np.array(det_texts, dtype=object)

    return det_boxes, det_texts
