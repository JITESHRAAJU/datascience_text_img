import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Basic settings
image_size = (128, 128)
batch_size = 32

# Load data from folders
datagen = ImageDataGenerator(rescale=1./255)

train = datagen.flow_from_directory(
    r'C:\Users\Win10\Desktop\muruga\train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')

test = datagen.flow_from_directory(
    r'C:\Users\Win10\Desktop\muruga\test',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Build a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(train.num_classes, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, epochs=5, validation_data=test)

# Evaluate model
loss, acc = model.evaluate(test)
print(f'\nTest Accuracy: {acc:.4f}, Loss: {loss:.4f}')

# Confusion matrix & report
y_true = test.classes
y_pred = np.argmax(model.predict(test), axis=1)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=test.class_indices.keys()))

# Confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test.class_indices, yticklabels=test.class_indices)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Predict one image
def predict_image(path):
    img = load_img(path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    label = list(test.class_indices.keys())[np.argmax(pred)]
    plt.imshow(img)
    plt.title(f'Predicted: {label}')
    plt.axis('off')
    plt.show()
    return label

# Example prediction
print("Predicted class:", predict_image(r"C:\Users\Win10\Desktop\muruga\train\cats\pexels-pixabay-45201.jpg"))
