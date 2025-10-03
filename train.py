from ultralytics import YOLO

model = YOLO("../DCD/ultralytics/cfg/models/DCD.yaml")


if __name__ == '__main__':

    model.train(data = "pcb_component.yaml", batch = 2, epochs = 100, imgsz = 640, project='DCD')

    #results = model.val()
    #predicts = model.predict("bus.jpg", save=True)

