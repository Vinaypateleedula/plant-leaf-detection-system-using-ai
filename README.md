# plant-leaf-detection-system-using-ai
pip install tensorflow opencv-python numpy matplotlib scikit-learn flask
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image_path, img_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, img_size)
    image = image.astype('float32') / 255.0
    return image

def load_dataset(dataset_path):
    # Implement loading and preprocessing dataset
    pass

# Example usage
image = preprocess_image('leaf_image.jpg')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model(input_shape=(128, 128, 3), num_classes=5):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.summary()
history = model.fit(train_images, train_labels, epochs=20, validation_data=(val_images, val_labels))
from flask import Flask, request, render_template
import cv2
import numpy as np

app = Flask(_name_)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        image = preprocess_image(file.stream)
        # Prediction code here
        # return the prediction result
        return 'Disease Detected'
    
if _name_ == "_main_":
    app.run(debug=True)
