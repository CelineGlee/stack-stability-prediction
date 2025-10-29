import tensorflow as tf
import pandas as pd
import os
import keras
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.transforms import RandomAffine, ColorJitter, RandomHorizontalFlip
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torchvision.models import ResNet18_Weights
import cv2
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0, ResNet50, VGG16
from tensorflow.keras.callbacks import  ModelCheckpoint
from sklearn.model_selection import KFold
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torchvision.models as models
from torchvision.models import ResNeXt101_32X8D_Weights

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print("Using GPU")
else:
    print("No GPU detected, using CPU")

# Set paths for the data
syspath = "/kaggle/input/cvproject/" # you need to replace in your syspath here
image_dir = f'{syspath}stack-stability-prediction/src/data/train/'  # Path to the folder containing the images
csv_path = syspath + "/train.csv"  # Path to the CSV file containing labels

# Load the CSV file with image ids and labels
data = pd.read_csv(csv_path)
data['id'] = data['id'].astype(str) + '.jpg'
data['stable_height'] -= 1

# Ensure labels are integers and within the correct range
data['stable_height'] = data['stable_height'].astype(int)  # Ensure labels are integers
# Split the data into train and validation sets
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)


# Function to create data generators with different augmentation scales
def create_data_generators(train_df, val_df, scale):
    flip = True

    if scale == 0:
        flip = False

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=flip,
        brightness_range=[1 - scale, 1 + scale],  # Adjust brightness
        zoom_range=[1 - scale, 1 + scale],  # Adjust zoom
        width_shift_range=scale,  # Adjust width shift
        height_shift_range=scale,  # Adjust height shift
        shear_range=scale,
        preprocessing_function=preprocess_input
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=image_dir,
        x_col='id',
        y_col='stable_height',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=image_dir,
        x_col='id',
        y_col='stable_height',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        shuffle=False
    )

    return train_generator, val_generator


# Define the model using EfficientNetB0
def create_model():
    # Base model
    base_model = EfficientNetB0(include_top=False, weights='imagenet',
                                input_shape=(224, 224, 3))
    inputs = layers.Input(shape=(224, 224, 3))

    # Fully connected layers
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Classification layer
    regression_output = layers.Dense(6, name='regression_output')(x)
    classification_output = layers.Softmax(name='classification_output')(regression_output)

    # Compiling the model
    model = models.Model(inputs=inputs, outputs=classification_output)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model


# Function to train the model with different augmentation scales
def train_model_with_scale(scale, epochs=10):
    train_generator, val_generator = create_data_generators(train_df, val_df, scale)
    smodel = create_model()

    # Train the model
    history = smodel.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )

    return history


# Train the model with different scales
scales = [0.0, 0.1, 0.2]  # No augmentation, ±0.1 scale, ±0.2 scale
histories = []

for scale in scales:
    print(f"\nTraining with scale: {scale}")
    history = train_model_with_scale(scale)
    histories.append(history)

# Initialize lists to store best accuracy and loss for each scale
best_acc = []
best_loss = []

# Extract best accuracy and loss from each history
for history in histories:
    best_acc.append(max(history.history['val_acc']))  # Best validation accuracy
    best_loss.append(min(history.history['val_loss']))  # Best validation loss

# Create a bar plot for best accuracy
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.bar(np.arange(len(scales)), best_acc, color='skyblue')
plt.xticks(np.arange(len(scales)), [f'Scale {s}' for s in scales])
plt.title('Best Validation Accuracy by Data Augmentation Scale')
plt.ylabel('Accuracy')
plt.xlabel('Data Augmentation Scale')

# Plot loss
plt.subplot(1, 2, 2)
plt.bar(np.arange(len(scales)), best_loss, color='lightcoral')
plt.xticks(np.arange(len(scales)), [f'Scale {s}' for s in scales])
plt.title('Best Validation Loss by Data Augmentation Scale')
plt.ylabel('Loss')
plt.xlabel('Data Augmentation Scale')

plt.tight_layout()
plt.show()

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Create generators
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=image_dir,
    x_col='id',
    y_col='stable_height',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    directory=image_dir,
    x_col='id',
    y_col='stable_height',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

# Function to create models with different backbones
def create_model(base_model):
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(6),
        layers.Softmax()
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model

# List of pretrained models to experiment with
pretrained_models = {
    "EfficientNetB0": EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
    "ResNet50": ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
    "VGG16": VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
}

# Dictionary to store the results of each model
results = {}

# Loop over each pretrained model
for model_name, base_model in pretrained_models.items():
    print(f"Training model: {model_name}")

    # Create model
    model = create_model(base_model)

    # Set up model checkpointing
    model_checkpoint_path = f'best_model_{model_name}.keras'
    checkpoint_callback = ModelCheckpoint(
        model_checkpoint_path,
        monitor='val_acc',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Train the model
    num_epochs = 30
    history = model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=val_generator,
        callbacks=[checkpoint_callback]
    )

    # Save training history for this model
    results[model_name] = history.history

# Plot training vs validation accuracy and loss for each model
for model_name, history in results.items():
    # Create a figure for each model
    plt.figure(figsize=(14, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot for the current model
    plt.suptitle(f'Performance of {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate title
    plt.show()

# Function to create data generators with a fixed augmentation scale
def create_data_generators(train_df, val_df):
    scale = 0.2  # Fixed scale for augmentation
    flip = True

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=flip,
        brightness_range=[1 - scale, 1 + scale],  # Adjust brightness
        zoom_range=[1 - scale, 1 + scale],  # Adjust zoom
        width_shift_range=scale,  # Adjust width shift
        height_shift_range=scale,  # Adjust height shift
        shear_range=scale,
        preprocessing_function=preprocess_input
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=image_dir,
        x_col='id',
        y_col='stable_height',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=image_dir,
        x_col='id',
        y_col='stable_height',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        shuffle=False
    )

    return train_generator, val_generator


# Define the model using EfficientNetB0 with decreasing number of dense layers
def create_model(num_dense_layers):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)

    # Define sizes for the dense layers, decreasing from 512
    layer_sizes = [512, 256, 128, 64]  # Adjust sizes as needed

    for i in range(num_dense_layers):
        x = layers.Dense(layer_sizes[i], activation='relu')(x)
        x = layers.Dropout(0.3)(x)

    regression_output = layers.Dense(6, name='regression_output')(x)
    classification_output = layers.Softmax(name='classification_output')(regression_output)
    model = models.Model(inputs=inputs, outputs=classification_output)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model


# Function to train the model with K-Fold cross-validation
def cross_validate_model(num_dense_layers, k=5, epochs=10):
    kf = KFold(n_splits=k)
    best_accs = []
    best_losses = []

    print(f"\nCross-Validating with {num_dense_layers} dense layers")

    for fold, (train_index, val_index) in enumerate(kf.split(data)):
        print(f"\nFold {fold + 1}/{k}")
        train_df = data.iloc[train_index]
        val_df = data.iloc[val_index]

        train_generator, val_generator = create_data_generators(train_df, val_df)
        model = create_model(num_dense_layers)

        # Train the model and track the best accuracy and loss
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            verbose=1
        )

        # Track only the best validation accuracy and loss for this fold
        best_acc = max(history.history['val_acc'])
        best_loss = min(history.history['val_loss'])

        best_accs.append(best_acc)
        best_losses.append(best_loss)

    return best_accs, best_losses


# Test different configurations of dense layers
num_layers_options = [1, 2, 3, 4]  # Varying number of dense layers
avg_best_acc = []
avg_best_loss = []

# Cross-validate over different layer configurations
for num_layers in num_layers_options:
    best_accs, best_losses = cross_validate_model(num_layers)
    avg_best_acc.append(np.mean(best_accs))  # Average best accuracy across folds
    avg_best_loss.append(np.mean(best_losses))  # Average best loss across folds

# Create a line plot for average accuracy and loss
plt.figure(figsize=(14, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(num_layers_options)), avg_best_acc, marker='o', linestyle='-', color='skyblue')
plt.xticks(np.arange(len(num_layers_options)), [f'{num_layers} Layers' for num_layers in num_layers_options])
plt.title('Average Best Validation Accuracy by Dense Layers (Fixed Scale 0.2)')
plt.ylabel('Accuracy')
plt.xlabel('Number of Dense Layers')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(np.arange(len(num_layers_options)), avg_best_loss, marker='o', linestyle='-', color='lightcoral')
plt.xticks(np.arange(len(num_layers_options)), [f'{num_layers} Layers' for num_layers in num_layers_options])
plt.title('Average Best Validation Loss by Dense Layers (Fixed Scale 0.2)')
plt.ylabel('Loss')
plt.xlabel('Number of Dense Layers')

plt.tight_layout()
plt.show()

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale = 1.0/255,  # Normalize pixel values to [0, 1]
    horizontal_flip = True,  # Randomly flip images horizontally
    brightness_range = [0.8, 1.2],  # Adjust brightness by a random factor
    zoom_range = [0.8, 1.2],  # Randomly zoom into images
    width_shift_range = 0.2,  # Randomly shift images horizontally (10% of total width)
    height_shift_range = 0.2,  # Randomly shift images vertically (10% of total height)
    shear_range = 0.2,  # Apply random shear transformations
    preprocessing_function=preprocess_input  # Normalize using ImageNet mean and std
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                preprocessing_function=preprocess_input)

# Create generators
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=image_dir,
    x_col='id',
    y_col='stable_height',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',  # For regression, use 'raw'
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    directory=image_dir,
    x_col='id',
    y_col='stable_height',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',  # For regression, use 'raw'
    shuffle=False
)

# Define the model using EfficientNetB0
def create_model():
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Input layer
    inputs = layers.Input(shape=(224, 224, 3))

    # EfficientNetB0 as the base model
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)

    # Add layers to the model
    x = layers.GlobalAveragePooling2D()(base_model.output)  # Global average pooling
    x = layers.Dense(512, activation='relu')(x)              # Dense layer with ReLU
    x = layers.Dropout(0.3)(x)                               # Dropout layer
    x = layers.Dense(256, activation='relu')(x)              # Dense layer with ReLU
    x = layers.Dropout(0.3)(x)                               # Dropout layer
    x = layers.Dense(128, activation='relu')(x)              # Dense layer with ReLU
    x = layers.Dropout(0.3)(x)                               # Dropout layer

    # Outputs
    regression_output = layers.Dense(6, name='regression_output')(x)
    classification_output = layers.Softmax(name='classification_output')(regression_output)

    # Create the model
    model = keras.models.Model(inputs=inputs, outputs= classification_output)


    # Adjust optimizer and loss function for mixed precision
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model

# Create and train the model
model = create_model()

# Set the filepath to save the best model
model_checkpoint_path = 'best_model.keras'

# Set up ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    model_checkpoint_path,
    monitor='val_acc',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the model
num_epochs = 30
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator,
    callbacks=[checkpoint_callback],
)




best_model = load_model('best_model.keras')
# # Load the previously saved model

# # Load the test CSV file
test_csv_path = f'{syspath}stack-stability-prediction/src/data/test/test.csv'
test_data = pd.read_csv(test_csv_path)
test_data['img_id'] = test_data['id'].astype(str) + '.jpg'
# Set the directory for the test images
test_image_dir = f'{syspath}stack-stability-prediction/src/data/test/test'

# Create a data generator for test images
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Create a generator for the test data
test_generator = test_datagen.flow_from_dataframe(
    test_data,
    directory=test_image_dir,
    x_col='img_id',
    y_col=None,
    target_size=(224, 224),  # Adjust to match input size of the model
    batch_size=32,
    class_mode=None,
    shuffle=False  # Don't shuffle the test data
)


predictions = model.predict(test_generator)
predicted_classes = predictions.argmax(axis=1) + 1  # For classification

# Create a DataFrame with predictions
results = pd.DataFrame({
    'id': test_data['id'],  # Image IDs
    'predicted_stable_height': predicted_classes  # Predictions
})

# Save predictions to a CSV file
results.to_csv('predictions.csv', index=False)

best_model_path = "best_model.keras"

best_model = load_model(best_model_path)

# Get predictions
val_predictions = best_model.predict(val_generator)

#
predicted_classes = val_predictions.argmax(axis=1)  # For classification
val_preds = pd.DataFrame({"id": val_df['id'], "true":val_df['stable_height'], "pred":predicted_classes})


# Calculate errors (1 for incorrect prediction, 0 for correct)
val_preds['error'] = (val_preds['true'] != val_preds['pred']).astype(int)

# Group by the true height class and calculate the proportion of errors
error_by_height = val_preds.groupby('true')['error'].mean()

# Calculate the distribution of true heights (i.e., count of each height class)
height_distribution = val_preds['true'].value_counts().sort_index()

# Create subplots: 2 plots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Error rate for each height class
axes[0].bar(error_by_height.index, error_by_height.values, color='salmon')
axes[0].set_title('Proportion of Prediction Errors by Height Class')
axes[0].set_xlabel('True Height Class')
axes[0].set_ylabel('Error Rate')
axes[0].set_xticks(error_by_height.index)

# Plot 2: Distribution of true height classes
axes[1].bar(height_distribution.index, height_distribution.values, color='skyblue')
axes[1].set_title('Height Class Distribution')
axes[1].set_xlabel('Height Class')
axes[1].set_ylabel('Number of Samples')
axes[1].set_xticks(height_distribution.index)

# Adjust layout
plt.tight_layout()
plt.show()

cm = confusion_matrix(val_df['stable_height'], predicted_classes)

# Define class labels (in this case, 1 to 6)
class_labels = np.arange(1, 7)

# Create the heatmap with the appropriate class labels
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

for i in range(6):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

    print(f'Class {i}: Precision={Precision:.2f}, Recall={Recall:.2f}, F1 Score={F1:.2f}')

def make_gradcam_heatmap(img_array, model, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions

    grad_model = keras.models.Model(
        model.inputs, [model.get_layer("top_conv").output, model.get_layer("regression_output").output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def display_gradcam(img_name, heatmap, alpha=0.4):
    # Load the original image
    img = image.load_img(img_name)
    img = image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Display superimpose result
    plt.matshow(superimposed_img)
    plt.axis('off')
    plt.show()


inc = 0

for i in range(val_preds.shape[0]):
    if inc == 12:
        break

    if val_preds.iloc[i]['true'] != val_preds.iloc[i]['pred']:
        inc += 1
        img_id = val_preds.iloc[i]['id']

        # Load and preprocess your image
        img = keras.utils.load_img(image_dir + img_id)  # Path to your image
        img = keras.utils.img_to_array(img)
        # Normalize to [0, 1]
        img = img / 255.0  # Scale pixel values to [0, 1]
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)
        print("Image: ", val_preds.iloc[i]['id'])
        print("Predicted:", val_preds.iloc[i]['pred'] + 1)
        print("Correct:", val_preds.iloc[i]['true'] + 1)
        heatmap = make_gradcam_heatmap(img, model)

        display_gradcam(image_dir + img_id, heatmap)


# BlockStackDataset class for handling image loading and preprocessing
class BlockStackDataset(Dataset):
    def __init__(self, data_file, img_dir, transform=None):
        """
        Initializes the dataset with the data file (pandas DataFrame), image directory,
        and optional transformations.

        Args:
        - data_file: A pandas DataFrame containing image information and labels.
        - img_dir: Path to the directory containing the images.
        - transform: Optional transformations to be applied to the images (e.g., resizing).
        """
        self.data_file = data_file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data_file)

    def __getitem__(self, idx):
        """
        Retrieves an image, label, and instability type based on the given index.

        Args:
        - idx: Index of the item in the dataset to retrieve.

        Returns:
        - image: The processed image.
        - label: The stable height label.
        - instability_type: The instability type.
        """
        try:
            # Construct the image file path
            img_name = self.img_dir + '/' + str(self.data_file.iloc[idx, 0]) + '.jpg'
            image = Image.open(img_name).convert('RGB')

            # Retrieve the label from the DataFrame
            label = self.data_file.iloc[idx, 6] - 1
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return None


# Dataset class for handling test data
class TestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Initialize by loading the csv file and setting image directory and transformations
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        # Return the total number of data entries (images)
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the image filename based on the index
        img_name = self.img_dir + '/' + str(self.data.iloc[idx, 0]) + '.jpg'
        image = Image.open(img_name).convert('RGB')  # Open and convert image to RGB

        # Apply transformations if any are defined
        if self.transform:
            image = self.transform(image)

        # Return image and its corresponding ID from the csv file
        return image, self.data.iloc[idx, 0]


# Function to define and return data augmentation and preprocessing transformations
def Data_Augumentation():
    # Training transformations: resizing, flipping, color jitter, affine transformation, normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation transformations: resizing and normalization
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Test transformations: resizing and normalization
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform, test_transform


# Function to create a directory for logging and saving model data
def create_log_dir():
    # Get current time to create a unique directory for model training logs and results
    current_time = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    best_solution_dir = f'model_trained/experiment_{current_time}'

    # Create the directory if it does not exist
    os.makedirs(best_solution_dir, exist_ok=True)

    # Return the directory path
    return best_solution_dir


# Function to save model state and log training details
def save_model_and_log(model, optimizer, epoch, train_loss, val_accuracy, best_val_accuracy, log_dir):
    # Check if the current validation accuracy is the best
    is_best = val_accuracy > best_val_accuracy
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_accuracy': val_accuracy,
        'best_val_accuracy': best_val_accuracy,
    }

    torch.save(checkpoint, os.path.join(log_dir, 'last_model.pth'))

    # If the current model is the best, save it separately
    if is_best:
        torch.save(checkpoint, os.path.join(log_dir, 'best_model.pth'))
        print(f"New best model saved with accuracy: {val_accuracy:.4f}")

    # Log the training data to a text file
    log_data = f"Epoch: {epoch}, Train Loss: {train_loss}, Val Accuracy: {val_accuracy}, Best Val Accuracy: {best_val_accuracy}\n"

    with open(os.path.join(log_dir, 'training_log.txt'), 'a') as f:
        f.write(log_data)


# Training Loop function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    best_solution_dir = create_log_dir()

    print("Trainning Model...")

    best_val_accuracy = 0.0

    # Lists to store loss and accuracy values over epochs
    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    # Loop through each epoch
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        train_accuracy = 0.0

        # Visualize progress using tqdm
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")

            # Loop through batches of training data
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                # Reset gradients
                optimizer.zero_grad()

                # Forward pass
                raw_outputs = model(inputs)
                loss = criterion(raw_outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update loss and accuracy for the epoch
                train_loss += loss.item()
                _, predicted = torch.max(raw_outputs.data, 1)
                accuracy = (predicted == labels).sum().item() / labels.size(0)
                train_accuracy += accuracy

                # Display progress in tqdm
                tepoch.set_postfix(loss=train_loss / len(train_loader), accuracy=train_accuracy / len(train_loader))

            # Compute average loss and accuracy for the epoch
            epoch_loss = train_loss / len(train_loader)
            epoch_accuracy = train_accuracy / len(train_loader)
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

        # Validation phase
        print("Validating...")

        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        all_labels = []
        all_predictions = []
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)  # Forward pass
                # predicted = torch.argmax(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate average validation accuracy
            val_accuracy = correct / total
            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

        # Print epoch summary
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'{best_solution_dir}/best_model.pth')
        print(f"Best accuracy so far: {best_val_accuracy:.4f}")

        # Save model and log details
        save_model_and_log(model, optimizer, epoch, train_losses, val_accuracy, best_val_accuracy, best_solution_dir)

    # Return final model and performance metrics
    return model, best_val_accuracy, train_accuracies, val_accuracies, train_losses, val_losses


# Data paths
train_image_folder = f'{syspath}stack-stability-prediction/src/data/train/train'
test_image_folder = f'{syspath}stack-stability-prediction/src/data/test/test'

train_csv_path = f'{syspath}stack-stability-prediction/src/data/train/train.csv'
test_csv_path = f'{syspath}stack-stability-prediction/src/data/test/test.csv'

batch_size = 16
learning_rate = 0.001

# Data preparation and augumentation
train_transform, val_transform, test_transform = Data_Augumentation()

full_data = pd.read_csv(train_csv_path)

train_data, val_data = train_test_split(full_data, test_size=0.1, random_state=42, shuffle=False)
train_data = pd.concat([train_data] * 5, ignore_index=True)

print("Train data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)

# Create datasets with appropriate transforms
train_dataset = BlockStackDataset(train_data, train_image_folder, transform=train_transform)
val_dataset = BlockStackDataset(val_data, train_image_folder, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TestDataset(test_csv_path, test_image_folder, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class BlockStackDataset_imagePro(Dataset):
    def __init__(self, data_file, img_dir, transform=None):
        self.data_file = data_file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_file)

    def preprocess_image(self, image):
        # Convert the image to HSV for better color segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for all the block colors (adjust the values as needed)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        lower_red1 = np.array([0, 150, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 50])
        upper_red2 = np.array([180, 255, 255])

        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        lower_cyan = np.array([85, 150, 50])
        upper_cyan = np.array([95, 255, 255])

        lower_magenta = np.array([145, 150, 50])
        upper_magenta = np.array([155, 255, 255])

        # Create masks for each block color
        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
        mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask_red = mask_red1 | mask_red2  # Combine the two red masks
        mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        mask_cyan = cv2.inRange(hsv_image, lower_cyan, upper_cyan)
        mask_magenta = cv2.inRange(hsv_image, lower_magenta, upper_magenta)

        # Combine all masks to get the entire stack
        combined_mask = mask_blue | mask_red | mask_green | mask_yellow | mask_cyan | mask_magenta

        # Use the combined mask to extract the entire stack from the original image
        result_stack = cv2.bitwise_and(image, image, mask=combined_mask)

        return result_stack

    def __getitem__(self, idx):
        img_name = self.img_dir + '/' + str(self.data_file.iloc[idx, 0]) + '.jpg'

        # Read image using OpenCV
        image = Image.open(img_name).convert('RGB')

        # Apply preprocessing
        processed_image = self.preprocess_image(image)

        # Convert to PIL Image for PyTorch transforms
        processed_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        label = self.data_file.iloc[idx, 6] - 1

        if self.transform:
            processed_image = self.transform(processed_image)

        return processed_image, label


# load ResNet18 Model

print("Setting Model...")

# Load pre-trained ResNet18 and remove the last fully connected layer
model_resnet18 = models.resnet18(pretrained=True)
num_ftrs = model_resnet18.fc.in_features
model_resnet18.fc = nn.Linear(num_ftrs, 6)

# Define a custom fully connected layer for your specific task
# model_resnet18.fc = nn.Sequential(
#         nn.Linear(num_ftrs, 512),    # Adjust 512 according to your needs
#         nn.ReLU(),
#        nn.Dropout(0.5),
#        nn.Linear(512, 1)            # Output layer for regression (predicting stable height)
# )

# Model, loss, and optimizer
model_resnet18 = model_resnet18.to(device)

criterion = nn.CrossEntropyLoss()

# criterion = nn.MSELoss()
optimizer = optim.Adam(model_resnet18.parameters(), lr=learning_rate)

for param in model_resnet18.parameters():
    param.requires_grad = True

resnet18, best_accuracy, train_accuracies, val_accuracies, train_losses, val_losses = train_model(model_resnet18,
                                                                                                  criterion, optimizer,
                                                                                                  train_loader,
                                                                                                  val_loader,
                                                                                                  num_epochs=10)

# Create datasets with appropriate transforms
train_dataset_imgP = BlockStackDataset_imagePro(train_data, train_image_folder, transform=train_transform)
val_dataset_imgP = BlockStackDataset_imagePro(val_data, train_image_folder, transform=val_transform)

# Create data loaders
train_loader_imgP = DataLoader(train_dataset_imgP, batch_size=batch_size, shuffle=True)
val_loader_imgP = DataLoader(val_dataset_imgP, batch_size=batch_size, shuffle=False)

# load ResNet18 Model

print("Setting Model...")

# Load pre-trained ResNet18 and remove the last fully connected layer
model_resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model_resnet18.fc.in_features
model_resnet18.fc = nn.Linear(num_ftrs, 6)

model_resnet18 = model_resnet18.to(device)

criterion = nn.CrossEntropyLoss()

# criterion = nn.MSELoss()
optimizer = optim.Adam(model_resnet18.parameters(), lr=learning_rate)

for param in model_resnet18.parameters():
    param.requires_grad = True

resnet18_imgP, best_accuracy, train_accuracies, val_accuracies = train_model(model_resnet18, criterion, optimizer,
                                                                             train_loader_imgP, val_loader_imgP,
                                                                             num_epochs=10)

# Model definition using resnext101_32x8d pre-trained on ImageNet
model_resnext101 = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2)

# Get the number of input features to the FC layer
num_ftrs = model_resnext101.fc.in_features
model_resnext101.fc = nn.Linear(num_ftrs, 6)

# Model, loss, and optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_resnext101 = model_resnext101.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_resnext101.parameters(), lr=learning_rate)

# Fine-tuning: Unfreeze all the layers
for param in model_resnext101.parameters():
    param.requires_grad = True  # Unfreeze all layers for fine-tuning

model_resnext101, best_accuracy, train_accuracies, val_accuracies = train_model(model_resnext101, criterion, optimizer, train_loader, val_loader, num_epochs = 10)

# Model definition using EfficientNetB4 pre-trained on ImageNet
model_efficientnetb4 = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
num_ftrs = model_efficientnetb4.classifier[1].in_features
model_efficientnetb4.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 6)  # Changed to 6 output neurons for 6 classes
)

model_efficientnetb4 = model_efficientnetb4.to(device)

criterion = nn.CrossEntropyLoss()  # Changed to CrossEntropyLoss for classification
optimizer = optim.Adam(model_efficientnetb4.parameters(), lr=learning_rate)

# Fine-tuning: Unfreeze all the layers
for param in model_efficientnetb4.parameters():
    param.requires_grad = True  # Unfreeze all layers for fine-tuning

# Training
model_efficientnetb4, best_accuracy, train_accuracies, val_accuracies = train_model(
    model_efficientnetb4, criterion, optimizer, train_loader, val_loader, num_epochs=10
)

# Model definition using EfficientNetB4 pre-trained on ImageNet
model_efficientnetb4 = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
num_ftrs = model_efficientnetb4.classifier[1].in_features
model_efficientnetb4.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 6)  # Changed to 6 output neurons for 6 classes
)

model_efficientnetb4 = model_efficientnetb4.to(device)

criterion = nn.CrossEntropyLoss()  # Changed to CrossEntropyLoss for classification
optimizer = optim.Adam(model_efficientnetb4.parameters(), lr=learning_rate)

# Fine-tuning: Unfreeze all the layers
for param in model_efficientnetb4.parameters():
    param.requires_grad = True  # Unfreeze all layers for fine-tuning

# Training
model_efficientnetb4, best_accuracy, train_accuracies, val_accuracies = train_model(
    model_efficientnetb4, criterion, optimizer, train_loader_imgP, val_loader_imgP, num_epochs = 10)