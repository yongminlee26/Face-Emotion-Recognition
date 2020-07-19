# Face Emotion Recognition Project 

import pandas 
import numpy
import scipy.special # sigmoid function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image # image handling module

# Face Emotion Recognition CNN definition
class ConvolutionNetwork :
    # building CNN
    def __init__(self) :
        # initialize CNN
        self.model = tf.keras.Sequential()
        # 1. 1st convolution layer
        # relu activation funtion -> to avoid negative numbers
        self.model.add(layers.Convolution2D(16, (3,3), input_shape=(48,48,1), activation='relu')) # number of filters, filterShape, inputShape
        # 2. Maxpooling
        self.model.add(layers.MaxPooling2D(pool_size=(2,2)))
        # 3. 2nd convolution layer
        self.model.add(layers.Convolution2D(32, (3,3), activation='relu'))
        # 4. Maxpooling
        self.model.add(layers.MaxPooling2D(pool_size=(2,2)))
        # 5. 3rd convolution layer
        self.model.add(layers.Convolution2D(64, (3,3), activation='relu'))
        # 6. Maxpooling
        self.model.add(layers.MaxPooling2D(pool_size=(2,2)))
        # 7 flatten output of maxPooling - Flatten()
        self.model.add(layers.Flatten())
        # fully connected layer - Dense()
        # 8. input layer
        self.model.add(layers.Dense(units=512, activation='relu'))
        # 9. hidden layer
        self.model.add(layers.Dense(units=1024, activation='relu'))
        # 10. drop out
        self.model.add(layers.Dropout(0.3))   
        # 11. output layer - emtion 카테고리 범위 : 0~6 => output layer의 node 갯수 : 7
        self.model.add(layers.Dense(units=7, activation='sigmoid'))
        # 12. compile CNN
        # loss function(=cost function) calculates loss  -> to find best weights, have to find the lowest loss
        # optimizer -> update weights
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

    # training CNN
    def fit(self, inputs, targets, ep) :
        self.model.fit(x=inputs,y=targets, epochs=ep)

    # evaluate CNN
    def evaluate(self, inputs, targets) :
        self.model.evaluate(inputs, targets)
    
    # predict CNN : 라벨링이 되지 않은 inputs을 모델에 전달하여 output을 반환
    def predict(self, inputs) :
        model_output = self.model.predict(inputs)
        list_output = model_output.tolist()
        final_output = numpy.empty(len(model_output))
        for i in range(0,len(model_output)):
            final_output[i] = list_output[i].index(max(list_output[i])) 
        return final_output


### European Faces Dataset processing ###

# data load
df = pandas.read_csv("fer2013.csv")

# seperate dataFrame into training data and testing data
training_df = df.loc[(df['Usage']=='Training'),:]
testing_df =  df.loc[(df['Usage']=='Testing'),:]

# training data processing : seperate the training data into training-target and training-input
# training-target data
emotion_training_df = training_df['emotion']
training_targets = emotion_training_df.to_numpy() # fetch image pixel data to numpy array 

# training-input data 
pixels_training_df = training_df['pixels'] 
np_pixels_training_df = pixels_training_df.to_numpy() # fetch image pixel data to numpy array

# 문자열을 ' '을기준으로 나누어서 문자리스트로 변환
for i in range(0, np_pixels_training_df.size):
    np_pixels_training_df[i] = np_pixels_training_df[i].split(' ')

# 문자리스트를 배열로 변환
for i in range(0, np_pixels_training_df.size):
    np_pixels_training_df[i] = numpy.array(np_pixels_training_df[i])

# normalization
# 배열의 원소들의 type을 float32으로변환
for i in range(0, np_pixels_training_df.size):
    np_pixels_training_df[i] = np_pixels_training_df[i].astype('float32')
# array reshaping : 48x48 -> 48 x 48 x 1
training_inputs = numpy.empty( (np_pixels_training_df.size, np_pixels_training_df[0].size) ) 
for i in range(0, np_pixels_training_df.size):
        training_inputs[i] = np_pixels_training_df[i]
training_inputs = training_inputs.reshape(len(training_inputs), 48,48,1)
# 배열의 원소들을 rescaling
training_inputs = training_inputs/255


# testing data processing : seperate the testing data into testing-target and testing-input
# testing-target data
emotion_testing_df = testing_df['emotion']
testing_targets = numpy.empty(emotion_testing_df.size)
idx = 0
for i in emotion_testing_df:
    testing_targets[idx] = i # fetch image pixel data to numpy array 
    idx += 1

# testing-input-data
pixels_testing_df = testing_df['pixels'] 
np_pixels_testing_df = pixels_testing_df.to_numpy() # fetch image pixel data to numpy array 

# 문자열을 ' '을기준으로 나누어서 문자리스트로 변환
for i in range(0, np_pixels_testing_df.size):
    np_pixels_testing_df[i] = np_pixels_testing_df[i].split(' ')

# 문자리스트를 배열로 변환
for i in range(0, np_pixels_testing_df.size):
    np_pixels_testing_df[i] = numpy.array(np_pixels_testing_df[i])

# normalization
# 배열의 원소들의 type을 float32으로변환
for i in range(0, np_pixels_testing_df.size):
    np_pixels_testing_df[i] = np_pixels_testing_df[i].astype('float32')
# array reshaping : 48x48 -> 48 x 48 x 1
testing_inputs = numpy.empty( (np_pixels_testing_df.size, np_pixels_testing_df[0].size) ) 
for i in range(0, np_pixels_testing_df.size):
        testing_inputs[i] = np_pixels_testing_df[i]
testing_inputs = testing_inputs.reshape(len(testing_inputs), 48,48,1)
# 배열의 원소들을 rescaling
testing_inputs = testing_inputs/255


### training model with European Faces Dataset ###

# building CNN
cnn = ConvolutionNetwork()

# training CNN
epochs = 20
cnn.fit(training_inputs,training_targets,epochs)


### testing model ###

# evaluate CNN
cnn.evaluate(testing_inputs, testing_targets)


### Labeling korean faces Dataset ###

# read image file name
korean_face_dataset = pandas.read_csv("korean_face_dataset_12131569.csv")

# make image file path
img_path = korean_face_dataset.loc[:,'File']
img_path = img_path.to_numpy()
img_path = './Images/' + img_path

# number of images 
num_img = len(img_path) 

# images-inputs data processing
# pixel size는 48 x 48 이므로,
# 이미지 정보를 저장할 이미지갯수*48*48 크기의 배열을 만든다
images_inputs = numpy.empty(num_img*48*48) 
images_inputs = images_inputs.reshape(num_img,48,48) # reshape

for i in range (0, num_img):
    image = Image.open(img_path[i])
    image = image.resize((48,48))   # 이미지크기는 파일마다 다르므로, 모델의 input shape에 맞게 resizing
    image = image.convert('L')      # 흑백으로
    image = numpy.array(image)
    images_inputs[i] = image

# Normalization
# reshaping array 
images_inputs = images_inputs.reshape(num_img, 48, 48, 1)
# rescaling
images_inputs = images_inputs / 255

# images_inputs을 모델에 전달하여 output 도출
final_model_outputs = cnn.predict(images_inputs)

# output을 가지고 korean_face_dataset_12131569.csv 파일에 라벨링
for i in range(0, num_img):
    korean_face_dataset['Emotion'] = final_model_outputs.astype('int8')
    
korean_face_dataset.to_csv("korean_face_dataset_12131569.csv", header=True, index=False)
