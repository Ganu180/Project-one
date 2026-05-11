import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

data = []
labels = []

categories = ["with_mask", "without_mask"]

print("Loading images...")

for category in categories:
    path = os.path.join("dataset", category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        try:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (224, 224))
            image = image / 255.0

            data.append(image)
            labels.append(category)
        except:
            print("Error loading image:", img_path)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

trainX, testX, trainY, testY = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Training started...")

model.fit(
    trainX,
    trainY,
    validation_data=(testX, testY),
    epochs=10,
    batch_size=32
)

model.save("mask_detector.h5")

print("Model saved successfully as mask_detector.h5")