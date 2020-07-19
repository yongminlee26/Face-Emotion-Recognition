Face Emotion Recognition Project 

You will be given a public dataset FER2013 of human facial expressions, 
which consists of images of European faces and corresponding emotions (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). 
In addition, Korean face images will be provided, however, they are not labeled.  
 
You have to create a Facial Expression Classifier. 
First, it needs to be trained on the European Faces Dataset (FER2013) and tested on the Test set of FER2013. 
Later, you need to test the performance of the classifier on the provided Test set of Korean Faces Dataset. 
Play with data and perform manipulation to achieve high performance on both datasets.  
 
Dataset 
The Facial Expression Recognition 2013 (FER-2013) Dataset.
FER2013 is in .csv file (fer2013.csv), where each row is one example: first column is label/emotion, second - representation of image pixels, 
and the third is usage (whether the image is Training or Testing)

Korean Faces dataset. ​
The folder consists of images and a .csv file, which has image file names and corresponding labels that need to be filled by you
 
FER2013 is images with (48x48x1) dimension, 
however, Korean Dataset consists of images of different sizes (MxNx3).
Apply data manipulation techniques to scale Korean dataset to have the same dimension as European dataset.
You may also perform data augmentation if required.  

* 구현코드파일 이외의 파일은 파일크기제한 문제로 미업로드