from ultralytics import YOLO

model=YOLO()

model.train(data="C:/Users/navee/dress_code_detection/data/data.yaml",epochs=50)