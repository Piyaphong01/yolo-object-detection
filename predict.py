from ultralytics import YOLO
# import cv2

model = YOLO("yolov26n_custom.pt")

# model.predict(source = "0", show=True, save=False,
#               line_width = 2, save_crop=False, save_txt = False, show_labels = True ,
#                show_conf = True, classes=[0,1] )
 
model.export(format="onnx")
