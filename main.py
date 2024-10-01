# main.py

from models.nested_unet import nested_unet
from preprocess.data_preprocessing import preprocess
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import jaccard_score
import numpy as np

# Load your dataset
images, masks = ...  # Load images and masks here
X_train, X_test, y_train, y_test = preprocess(images, masks)

# Initialize the model
model = nested_unet()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=16, epochs=20, validation_split=0.2)

# Evaluate and save the model
model.save('models/model_weights/best_model.h5')
