import numpy as np
from PIL import Image
from mtcnn import MTCNN

# Initialize Face Detector
detector = MTCNN()

def load_and_preprocess_image(image: Image.Image):
    img_array = np.array(image)
    faces = detector.detect_faces(img_array)

    if not faces:
        print("No face detected. Using full image.")
        cropped_image = image
    else:
        x, y, w, h = faces[0]["box"]
        x, y = abs(x), abs(y)
        cropped_image = image.crop((x, y, x + w, y + h))

    cropped_image = cropped_image.resize((224, 224))
    img_array = np.array(cropped_image) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_image_undertone(image: Image.Image):
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    return img_array.flatten().reshape(1, -1)
