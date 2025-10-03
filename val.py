from ultralytics import YOLO

model = YOLO("weight file path")



if __name__ == '__main__':
    metrics = model.val(data = 'pcb_component.yaml', batch = 1, imgsz=640, iou=0.65)  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    #metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category

