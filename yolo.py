import torch
import cv2
import numpy as np

def run_yolov5(image_path):
    model=torch.hub.load('ultralytics/yolov5','yolov5m',pretrained=True)
    image=cv2.imread(image_path)
    
    scale_factor=2
    image=cv2.resize(image,None,fx=scale_factor,fy=scale_factor)
    image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results=model(image_rgb)
    
    labels,coords=results.xyxyn[0][:,-1],results.xyxyn[0][:,:-1]
    class_names=model.names
    confidence_threshold = 0.3
    
    for i in range(len(labels)):
        confidence=coords[i][4].item()
        if confidence>=confidence_threshold:
            x_min,y_min,x_max,y_max=coords[i][:4]
            x_min,y_min,x_max,y_max=int(x_min*image.shape[1]),int(y_min*image.shape[0]),int(x_max*image.shape[1]),int(y_max*image.shape[0])
            class_label=class_names[int(labels[i])]

            cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(30,20,210),2)
            label_text=f"{class_label}:{confidence:.2f}"
            font_scale=0.65  
            thickness=2    
            cv2.putText(image,label_text,(x_min,y_min-10),cv2.FONT_HERSHEY_SIMPLEX,font_scale,(200,200,0),thickness)

    cv2.imshow("YOLOv5 Object Detection",image)
    cv2.waitKey(0)

image_path="/home/reynash/Downloads/download.jpeg"
run_yolov5(image_path)
