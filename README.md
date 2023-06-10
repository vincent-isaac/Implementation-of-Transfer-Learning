# Implementation-of-Transfer-Learning
## Aim:
To Implement Transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture.
## Problem Statement and Dataset:
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each:


VGG19 is a variant of the VGG model which in short consists of 19 layers (16 convolution layers, 3 Fully connected layer, 5 MaxPool layers, and 1 SoftMax layer).

Now we have used transfer learning with the help of VGG-19 architecture and use it to classify the CIFAR-10 Dataset
</br>
</br>
</br>

## DATA SET

![image](https://github.com/vincent-isaac/Implementation-of-Transfer-Learning/assets/75234588/1f610a44-a0d3-47dc-a44d-3375a1c898a2)


## DESIGN STEPS:

### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Load CIFAR-10 Dataset & use Image Data Generator to increse the size of dataset

### STEP 3:
Import the VGG-19 as base model & add Dense layers to it

### STEP 4:
Compile and fit the model

### Step 5:
Predict for custom inputs using this model

## PROGRAM:
NAME : VINCENT ISAAC JEYARAJ J </BR>
REG NO :212220230060
```python
import pandas as pd
import numpy as np
from keras import layers
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from keras import Sequential
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from tensorflow.keras import utils
from sklearn.metrics import classification_report,confusion_matrix

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

base_model=VGG19(
    include_top=False,
    weights='imagenet',
    input_shape=(32,32,3)
)


for layer in base_model.layers:
  layer.trainable = False
  
model=Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(800,activation=('relu')))
model.add(Dense(650,activation=('relu')))
model.add(Dropout(0.3))
model.add(Dense(500,activation=('relu')))
model.add(Dense(350,activation=('relu')))
model.add(Dense(200,activation=('relu')))
model.add(Dense(110,activation=('relu')))
model.add(Dropout(0.4))
model.add(Dense(98,activation=('relu')))
model.add(Dense(40,activation=('relu')))
model.add(Dense(10,activation=('softmax')))


model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
                                         
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test), callbacks=[learning_rate_reduction])

metrics = pd.DataFrame(model.history.history)

metrics[['loss','val_loss']].plot()

metrics[['accuracy','val_accuracy']].plot()

x_test_predictions = np.argmax(model.predict(x_test), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

```


## OUTPUT:
### Training Loss, Validation Loss Vs Iteration Plot:
![image](https://github.com/SanjayKumarAIML/Implementation-of-Transfer-Learning/assets/93427246/ed5fa7f9-82b3-4a96-974d-dbedebf48b0b)
</br>
![image](https://github.com/SanjayKumarAIML/Implementation-of-Transfer-Learning/assets/93427246/f9607dbb-83df-4799-87af-0ed45b62e551)
</br>
### Classification Report:
![image](https://github.com/SanjayKumarAIML/Implementation-of-Transfer-Learning/assets/93427246/16307c30-83c9-4306-bbc9-e29a54240ed9)
</br>

### Confusion Matrix:
![image](https://github.com/SanjayKumarAIML/Implementation-of-Transfer-Learning/assets/93427246/3dde1f3e-4910-4f62-a2e5-8bef9f8bd581)

</br>

## Conculsion:
* We got an Accuracy of 60% with this model.There could be several reasons for not achieving higher accuracy. Here are a few possible explanations:
### Dataset compatibility: 
* VGG19 was originally designed and trained on the ImageNet dataset, which consists of high-resolution images. 
* In contrast, the CIFAR10 dataset contains low-resolution images (32x32 pixels). 
* The difference in image sizes and content can affect the transferability of the learned features. 
* Pretrained models like VGG19 might not be the most suitable choice for CIFAR10 due to this disparity in data characteristics.

### Inadequate training data: 
* If the CIFAR10 dataset is relatively small, it may not provide enough diverse examples for the model to learn robust representations. 
* Deep learning models, such as VGG19, typically require large amounts of data to generalize well. 
* In such cases, you could consider exploring other architectures that are specifically designed for smaller datasets, or you might want to look into techniques like data augmentation or transfer learning from models pretrained on similar datasets.

### Model capacity: 
* VGG19 is a deep and computationally expensive model with a large number of parameters. 
* If you are limited by computational resources or working with a smaller dataset, the model's capacity might be excessive for the task at hand. 
* In such cases, using a smaller model architecture or exploring other lightweight architectures like MobileNet or SqueezeNet could be more suitable and provide better accuracy.

## RESULT:
Thus, transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture is successfully implemented.
