# mobilenetv2-transfer-learning

## Table of Contents
1. [Introduction](#introduction)
2. [Installation Instructions](#installation-instructions)
    - [Prerequisites](#prerequisites)
    - [Dependencies](#dependencies)
3. [Usage](#usage)
    - [Example Notebook Workflow](#example-notebook-workflow)
4. [Features](#features)
5. [Configuration](#configuration)

<a name="introduction"></a>
## 1. Introduction

This project demonstrates how to perform transfer learning using the MobileNetV2 architecture in TensorFlow/Keras. Transfer learning leverages pre-trained models, which have been trained on a large dataset, for tasks on a new dataset with limited data. The MobileNetV2 model is particularly suitable for mobile and embedded vision applications due to its efficiency in terms of speed and size.

The main steps include loading the MobileNetV2 model, freezing layers, adding custom classification layers, and fine-tuning the model. The dataset used in this example is assumed to be a collection of images categorized into classes, split into training and validation subsets.

<a name="installation-instructions"></a>
## 2. Installation Instructions

### Prerequisites
To get started with this project, ensure that you have the following installed:
- Python 3.7+
- Jupyter Notebook
- TensorFlow 2.x
- A GPU-compatible device (optional but recommended for faster training)

### Dependencies
The project relies on several Python libraries, which you need to install before running the notebooks. These include:
- TensorFlow: `pip install tensorflow`
- Matplotlib: `pip install matplotlib`
- NumPy: `pip install numpy`

To install all dependencies at once, you can use the following command:

```bash
pip install tensorflow matplotlib numpy
```

<a name="usage"></a>
## 3. Usage

This section explains how to use the Jupyter notebooks provided in this repository to perform transfer learning with the MobileNetV2 model. The following steps outline a typical workflow:

### Example Notebook Workflow

1. **Load the Dataset**:
    - The dataset should be organized into directories where each directory corresponds to a class.
    - Load the dataset using `image_dataset_from_directory()` from `tensorflow.keras.preprocessing`.

    Example:
    ```python
    train_dataset = image_dataset_from_directory(directory,
                                                 shuffle=True,
                                                 batch_size=32,
                                                 image_size=(160, 160),
                                                 validation_split=0.2,
                                                 subset='training',
                                                 seed=42)
    validation_dataset = image_dataset_from_directory(directory,
                                                 shuffle=True,
                                                 batch_size=32,
                                                 image_size=(160, 160),
                                                 validation_split=0.2,
                                                 subset='validation',
                                                 seed=42)
    ```

2. **Data Augmentation and Preprocessing**:
    - Perform data augmentation to artificially expand the dataset and improve model generalization.
    - Preprocess the input data using `mobilenet_v2.preprocess_input`.

3. **Load the Pre-trained MobileNetV2 Model**:
    - Load the MobileNetV2 model with pre-trained weights from ImageNet.
    - Set `include_top=False` to exclude the final classification layers.
    - Freeze the base model to retain the pre-trained weights during initial training.

    Example:
    ```python
    base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    ```

4. **Add Custom Classification Layers**:
    - Add global average pooling, dropout, and a dense layer with a single unit for binary classification.

    Example:
    ```python
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    ```

5. **Compile and Train the Model**:
    - Compile the model using the Adam optimizer and binary cross-entropy loss function.
    - Train the model using the training dataset and validate it with the validation dataset.

    Example:
    ```python
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_dataset,
                        validation_data=validation_dataset,
                        epochs=5)
    ```

6. **Fine-Tune the Model**:
    - Unfreeze the base model and fine-tune specific layers by setting them to `trainable=True`.
    - Reduce the learning rate for fine-tuning.

    Example:
    ```python
    base_model.trainable = True
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(lr=0.00001),
                  metrics=['accuracy'])

    history_fine = model.fit(train_dataset,
                             validation_data=validation_dataset,
                             epochs=5,
                             initial_epoch=history.epoch[-1])
    ```

7. **Evaluate the Model**:
    - Evaluate model performance on training and validation sets.
    - Plot the accuracy and loss to visualize training progress.

<a name="features"></a>
## 4. Features

- **Transfer Learning**: Utilize the MobileNetV2 model pre-trained on ImageNet for efficient and effective training on a new dataset.
- **Data Augmentation**: Enhance model performance by augmenting the dataset with various transformations.
- **Layer Freezing and Fine-Tuning**: Flexibility to freeze and unfreeze layers, allowing for effective fine-tuning of the model.
- **Custom Classification**: Tailor the model's output to specific tasks by adding custom classification layers.

<a name="configuration"></a>
## 5. Configuration

The following aspects of the project can be configured to fit specific use cases:

- **Dataset**: Modify the dataset path and structure as needed. Ensure that images are organized by class in separate directories.
- **Batch Size and Image Size**: Adjust the `BATCH_SIZE` and `IMG_SIZE` to optimize for memory and processing power.
- **Data Augmentation**: Customize the data augmentation pipeline by adding or modifying transformations such as flipping, rotation, zoom, etc.
- **Model Architecture**: Modify the MobileNetV2 architecture by unfreezing additional layers, changing the number of units in the classification layer, or adding other custom layers.
- **Learning Rate and Optimizer**: Experiment with different learning rates and optimizers to improve model convergence and performance.


