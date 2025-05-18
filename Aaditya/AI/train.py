from ultralytics import YOLO

def train_yolo():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Train the model using your custom dataset
    results = model.train(
        data='/Users/aaditya/Desktop/PESPECTIV/data.yaml',  # Path to data config file
        epochs=75,              # Number of epochs
        imgsz=640,              # Image size
        batch=4  # Name of the experiment
    )

if __name__ == "__main__":
    train_yolo()