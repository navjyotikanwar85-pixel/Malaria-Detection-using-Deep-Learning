import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

parasitized_path = "Parasitized"
uninfected_path = "Uninfected"

data = []
labels = []

# Load Parasitized images
for img in os.listdir(parasitized_path):
    path = os.path.join(parasitized_path, img)
    image = cv2.imread(path)

    if image is not None:
        image = cv2.resize(image,(64,64))
        image = image/255.0
        data.append(image)
        labels.append(0)

# Load Uninfected images
for img in os.listdir(uninfected_path):
    path = os.path.join(uninfected_path, img)
    image = cv2.imread(path)

    if image is not None:
        image = cv2.resize(image,(64,64))
        image = image/255.0
        data.append(image)
        labels.append(1)

data = np.array(data)
labels = np.array(labels)

print("Dataset:",data.shape)

X_train,X_test,y_train,y_test = train_test_split(
data,labels,test_size=0.2,random_state=42
)

# CNN Model
model = Sequential([

Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
MaxPooling2D(2,2),

Conv2D(64,(3,3),activation='relu'),
MaxPooling2D(2,2),

Conv2D(128,(3,3),activation='relu'),
MaxPooling2D(2,2),

Flatten(),

Dense(128,activation='relu'),
Dropout(0.5),

Dense(2,activation='softmax')

])

model.compile(
optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)

early = EarlyStopping(patience=3)

history = model.fit(
X_train,y_train,
epochs=10,
validation_data=(X_test,y_test),
callbacks=[early]
)

loss,acc = model.evaluate(X_test,y_test)

print("Accuracy:",acc)

y_pred = np.argmax(model.predict(X_test),axis=1)

print(classification_report(y_test,y_pred))

model.save("malaria_model.h5")

print("Model saved successfully")