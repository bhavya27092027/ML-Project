# Learn Hand-Written Digits

import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()       #x_train has image of number and y_train has its numeric value taken from this dataset
print(len(x_train))
print(len(x_test))

print(x_train.shape)   
print(y_train.shape)   

print(x_train[0])
print(y_train[0])

plt.matshow(x_train[0])
#plt.show()

# Scaling and converting integer values in between 0 and 1 for machine to perform calculations easily.
x_train=x_train/255
x_test=x_test/255

# total inputs=28*28 means 784 features and for giving these features we will convert 2-D data to a vector using reshape
x_train_flattened=x_train.reshape(len(x_train),28*28)
print(x_train_flattened)
print(x_train_flattened.shape)
x_test_flattened=x_test.reshape(len(x_test),28*28)
print(x_test_flattened)
print(x_test_flattened.shape)

model=keras.Sequential(
    [
        keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')
    ]
)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train_flattened,y_train,epochs=5)

y_pred=model.predict(x_test_flattened)
print(y_pred)
print(np.argmax(y_pred[0]))
plt.matshow(x_test[0])
#plt.show()

y_predictions_labels=[np.argmax(i) for i in y_pred]
y_predictions_labels[ :5]          #first 5 predictions
cm=tf.math.confusion_matrix(labels=y_test,predictions=y_predictions_labels)      # how many times which number is correctly predicted or incorrectely predected and checking through accuracy

#creating hidden layers and accuracy increases
model=keras.Sequential(
    [
        keras.layers.Dense(100,input_shape=(784,),activation='relu'),
        keras.layers.Dense(10,activation='sigmoid')
    ]
)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train_flattened,y_train,epochs=5)

model.evaluate(x_test_flattened,y_test)   #checking accuracy on test data by giving output 

import seaborn as sn 
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()