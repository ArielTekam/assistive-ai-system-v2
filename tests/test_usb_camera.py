import cv2

cap = cv2.VideoCapture(0)

print("Camera ouverte:", cap.isOpened())

ret, frame = cap.read()
print("Lecture image:", ret)

if ret:
    cv2.imwrite("test_pi5_camera.jpg", frame)
    print("Image sauvegardée: test_pi5_camera.jpg")

cap.release()

