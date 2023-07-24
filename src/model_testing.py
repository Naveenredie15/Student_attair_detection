#load the image from any location
from ultralytics import YOLO

model=YOLO(r'C:\Users\navee\dress_code_detection\runs\detect\train6\weights\best.pt')

#sample image testing
path=r'C:\Users\navee\dress_code_detection\testimage.jpg'

model.predict(source=path)