import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

model = load_model("gesture_model.h5")

classes = sorted(os.listdir("dataset/train"))

# 👉 ALWAYS build safe path
base_dir = os.path.dirname(__file__)  # current file location

image_path = os.path.join(
    base_dir,
    "dataset/test/01_palm/01_palm_24.jpg"
)

print("Using image:", image_path)

img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found!")
    print("Check path above 👆")
    exit()

img_resized = cv2.resize(img, (64,64)) / 255.0
img_resized = np.expand_dims(img_resized, axis=0)

pred = model.predict(img_resized)

index = np.argmax(pred)
label = classes[index]

print("Prediction:", label)

cv2.putText(img, label, (10,40),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0,255,0), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()