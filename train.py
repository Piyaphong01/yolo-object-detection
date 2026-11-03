from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

model.train(
    data="dataset_custom.yaml",  
    imgsz=640,
    batch=8,
    epochs=100,                   
    workers=0,
    device=0
)