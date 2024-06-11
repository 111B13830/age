import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# 構建模型
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # 性別辨識（二分類：0 或 1）
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 數據預處理和增強
def preprocess_data():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        'path/to/dataset',  # 替換成您的數據集路徑
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        'path/to/dataset',  # 替換成您的數據集路徑
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    return train_generator, validation_generator

# 訓練模型
def train_model(model, train_generator, validation_generator):
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )

# 預測性別
def predict_gender(model, img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "Male"
    else:
        return "Female"

# 主程序
if __name__ == "__main__":
    model = build_model()
    train_generator, validation_generator = preprocess_data()
    train_model(model, train_generator, validation_generator)

    # 測試單張圖片
    img_path = 'path/to/test/image.jpg'  # 替換成您要測試的圖片路徑
    gender = predict_gender(model, img_path)
    print(f"The predicted gender is: {gender}")
