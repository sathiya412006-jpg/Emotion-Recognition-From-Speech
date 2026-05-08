import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical


DATASET_PATH = "ravdess_data"


emotion_labels = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}


def extract_features(file_path):

    try:

        audio, sample_rate = librosa.load(
            file_path,
            duration=3,
            offset=0.5
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc_scaled = np.mean(mfcc.T, axis=0)

        return mfcc_scaled

    except Exception as e:

        print("Error processing file:", file_path)
        print(e)

        return None


X = []
y = []

print("Loading Dataset...\n")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            emotion_code = file.split("-")[2]

            emotion = emotion_labels.get(emotion_code)

            features = extract_features(file_path)

            if features is not None:

                X.append(features)
                y.append(emotion)


print("Dataset Loaded Successfully\n")

print("Total Audio Files:", len(X))


X = np.array(X)
y = np.array(y)


encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42
)


X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)


model = Sequential()

model.add(
    LSTM(
        128,
        input_shape=(40, 1),
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(
    LSTM(
        64,
        return_sequences=False
    )
)

model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(
    y_categorical.shape[1],
    activation='softmax'
))


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


model.summary()


history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)


loss, accuracy = model.evaluate(X_test, y_test)

print(f"Accuracy : {accuracy * 100:.2f}%")


y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)


print(
    classification_report(
        y_true,
        y_pred_classes,
        target_names=encoder.classes_
    )
)


print(confusion_matrix(y_true, y_pred_classes))


plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Training Accuracy', 'Validation Accuracy'])

plt.show()


sample_prediction = encoder.inverse_transform(
    [y_pred_classes[0]]
)

print("Predicted Emotion :", sample_prediction[0])