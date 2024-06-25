import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.nasnet import NASNetLarge
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0],True)
img_size = (331, 331)
img_full_size = (331,331,3)
batchsize = 8

allimages = np.load("/Users/yaremayaremcuk/Documents/data/allimages.npy")
alllabels = np.load("/Users/yaremayaremcuk/Documents/data/alllabels.npy")

print(allimages.shape)
print(alllabels.shape)


allimagesmod = allimages / 255.0

le = LabelEncoder()
integerlables = le.fit_transform(alllabels)




numofcateg = len(list(le.classes_))



alllabelsmod = to_categorical(integerlables, num_classes=numofcateg)
print(alllabelsmod)


print("Splitting into train and test")

X_train, X_test, y_train, y_test = train_test_split(allimagesmod, alllabelsmod, test_size=0.3, random_state=42)

print("X_train, X_test, Y_train, Y_test ------> shapes: ")

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)



del allimages
del alllabels
del integerlables
del allimagesmod
del alllabelsmod


mymodel = NASNetLarge(input_shape=img_full_size , weights='imagenet', include_top=False)

for layer in mymodel.layers:
    layer.trainable = False
    print(layer.name)

plusflattenlayer = Flatten()(mymodel.output)

predict = Dense(numofcateg, activation='softmax')(plusflattenlayer)

model = Model(inputs = mymodel.input, outputs = predict)


learning_rate = 1e-4
opt = Adam(learning_rate)

model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)


stepsperepoch = np.ceil(len(X_train) / batchsize)
validationsteps = np.ceil(len(X_test) / batchsize)



fin_model = "/Users/yaremayaremcuk/Documents/dogfind/dogsfindmodel.h5"

callbacks = [
    ModelCheckpoint(fin_model, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy', patience=7, verbose=1)
]

with tf.device('/cpu:0'):    
    X_t = tf.convert_to_tensor(X_train, np.float32)    
    Y_t = tf.convert_to_tensor(y_train, np.float32)
    X_vt = tf.convert_to_tensor(X_test, np.float32)    
    Y_vt = tf.convert_to_tensor(y_test, np.float32)




r = model.fit(
x=X_t, y=Y_t,
validation_data=(X_vt,Y_vt),
epochs= 30,
batch_size=batchsize,
steps_per_epoch=stepsperepoch,
validation_steps=validationsteps,
callbacks=[callbacks]
)




acc = r.history['accuracy']

val_acc = r.history['val_accuracy']
loss = r.history['loss']
val_loss = r.history[ 'val_loss']
epochsForPlot = range(len(acc))



plt.plot(epochsForPlot, acc, 'r' , label= 'Train Accuracy') 
plt.plot(epochsForPlot, val_acc, 'b' , label= 'Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(epochsForPlot, loss, 'r' , label= 'Train Loss') 
plt.plot(epochsForPlot, val_loss, 'b' , label= 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')
plt.show()