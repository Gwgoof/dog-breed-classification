## End-to-end Multi-class Dog Breed Classification Project
In this project I use machine learning and transfer learning to identify different breeds of dogs from pictures. To do this I built an end-to-end multi-class image classifier using TensorFlow 2.x and TensorFlow Hub.

**Fun Fact:** Multi-class image classification is the kind of technology Tesla uses in their self-driving cars and Airbnb uses to automatically add features and amenities to their listings based on imagery.

### The Environment
- Python 3
- Pandas, Numpy, Matplotlib, Seaborn
- TensorFlow 2.x
- TensorFlow Hub
- GPU

### 1. Problem Definition
Identifying the breed of any given image of a dog.

### 2. Data
The data used was from the dog breed identification competition on [Kaggle](https://www.kaggle.com/c/dog-breed-identification/data).

### 3. Evaluation
[The evaluation](https://www.kaggle.com/c/dog-breed-identification/overview/evaluation) is a file with prediction probabilities for each dog breed of each test image.

### 4. Features
Some information about the data:

- These are images (unstructured data) so it's probably best to use deep learning/transfer learning.
- There are 120 breeds of dogs (this means there are 120 different classes).
- There are around 10,000+ images in the training set (these images have labels).
- There are around 10,000+ images in the test set (these images have no labels, because I need to predict them).

### 5. Modeling
For the machine learning model, I used a pretrained [`mobilenet_v2_130`](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4) deep learning model from TensorFlow Hub. I also created and included two callbacks, one for TensorBoard which helps track the models progress and another for early stopping which prevents the model from training for too long. I first trained the model on 1000 images, to make sure everything is working fine, before opening it up to the full dataset, to then make and evaluate predictions.
