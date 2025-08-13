from ultralytics import YOLO
import cv2

model = YOLO('models/yolov8m-seg.pt')
img = cv2.imread('data/generated/generatedRGB/000084.png')
results = model(img)

# results é uma lista; pegue o primeiro elemento
result = results[0]

# Para visualizar as máscaras e bounding boxes:
result.plot()  # Isso retorna uma imagem com as detecções desenhadas

# Exiba com OpenCV
cv2.imshow("Detecção", result.plot())
cv2.waitKey(0)
cv2.destroyAllWindows()

# Veja as classes detectadas
print(result.boxes.cls)