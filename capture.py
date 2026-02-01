import cv2
import os

label = "A"
save_dir = f"data/raw/{label}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    cv2.imshow("Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f"{save_dir}/{count}.jpg", frame)
        count += 1
        print("Saved:", count)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
