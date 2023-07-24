from ultralytics import YOLO
import cv2


model=YOLO(r'C:\Users\navee\dress_code_detection\runs\detect\train6\weights\best.pt')

# 0 for taking video from webcam or else we can manually give path
video_path=r'C:\Users\navee\dress_code_detection\Snapchat-663660182.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():

    success, frame = cap.read()

    if success:
        results = model(frame,save_txt=True)
        
        annotated_frame = results[0].plot()
        
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
