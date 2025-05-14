import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

#Read the data
X_full = pd.read_csv('./data/train.csv')
X_test_full = pd.read_csv('./data/test.csv')

#Clean and split the data
column_diff = X_full.columns.difference(X_test_full.columns)
y_label = list(column_diff)[0]
features = list(X_test_full.columns)

y = X_full[y_label].copy()
X = X_full[features].copy()

y = y.values

y = to_categorical(y,10)

X = X.values

X = X.reshape(-1, 28, 28, 1)

X = X / 255.0

X_test_full = X_test_full.values
X_test_full = X_test_full.reshape(-1, 28, 28, 1)
X_test_full = X_test_full / 255.0

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

X_train,X_cv,Y_train,Y_cv = train_test_split(X,y,test_size=0.2,random_state=42)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(
    x=X_train,
    y=Y_train,
    validation_data=(X_cv, Y_cv),
    epochs=100,
    callbacks=[callback]
)

predictions = model.predict(X_test_full)
predicted_classes = np.argmax(predictions, axis=1)

# Create a DataFrame
submission_df = pd.DataFrame({
    "ImageId": np.arange(1, len(predicted_classes) + 1),
    "Label": predicted_classes
})

# Save to CSV without the index
submission_df.to_csv("submission.csv", index=False)

print("Submission file saved successfully!")