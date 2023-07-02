import cv2
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("error")
        break
    else:
        results = model(frame)
        img = results[0].plot()
        cv2.imshow("yolov8 test",img)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


