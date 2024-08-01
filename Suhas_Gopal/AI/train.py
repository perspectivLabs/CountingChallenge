from ultralytics import YOLO


def main():
    model = YOLO("checkpoints/yolov8n.pt")

    training_options = {
        'batch': 4,
        'plots': True,
        'amp': False,
        'lr0': 0.005,
        'single_cls': True,
        'name': "object_counting",
        'imgsz': 1280,
        'flipud': 0.3,
        'mixup': 0.3
    }
    model.train(data="data.yaml", epochs=100, **training_options)




if __name__ == '__main__':
    main()