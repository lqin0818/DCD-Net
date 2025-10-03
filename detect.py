from ultralytics import YOLO

model = YOLO("weight file path")
source = "image folder or file path"


if __name__ == '__main__':
    model.predict(source, save=True, imgsz=640, line_width=1)
