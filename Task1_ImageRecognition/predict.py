import numpy as np
from PIL import Image
import tf_keras as keras

# تحميل المودل
model = keras.models.load_model("keras_model.h5", compile=False)

# قراءة أسماء الكلاسات
class_names = open("labels.txt", "r").readlines()

# اسم صورة الاختبار
image = Image.open("فون.jfif").convert("RGB")
# تجهيز الصورة
image = image.resize((224, 224))
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array

# التنبؤ
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]

print("Prediction:", class_name)
print("Confidence:", confidence_score)