import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator

# Postavljanje parametara za lakse promjene osjetljivih vrijednosti
path = "myData" # folder sa svim klasama slika
labelFile = 'oznake.csv' # .csv datoteka sa svim nazivima znakova
batch_size_val=50  # how many to process together
steps_per_epoch_val=2000
epochs_val=2 # koliko epoha ce se izvrsiti
dimension = (32,32,3) # dimenzije slika kojima se trenira model
testRatio = 0.2    # udio slika koje se koriste za testiranje (20%)
validationRatio = 0.2 # udio slika koje se koriste za validaciju (20% od 80%)

############################### Importing of the Images
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Ukupno klasa:", len(myList))
noOfClasses=len(myList)
print("Ucitavanje klasa...")
for x in range (0, len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
images = np.array(images)
classNo = np.array(classNo)
 
# Raspodjela podataka, gdje je X_train niz slika koje treba istrenirati, y_train njima odgovarajuci ID klase
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
steps_per_epoch_val = len(X_train)

# Provjera uvjeta stabilnosti modela, osigurava se uparivanje fotografija i oznaka i tocnosti dimenzija svih fotografija
assert(X_train.shape[0] == y_train.shape[0]), "Broj slika nije jednak broju oznaka u trening setu podataka"
assert(X_validation.shape[0] == y_validation.shape[0]), "Broj slika nije jednak broju oznaka u validacijskom setu podataka"
assert(X_test.shape[0] == y_test.shape[0]), "Broj slika nije jednak broju oznaka u testnom setu podataka"
assert(X_train.shape[1:] == dimension), "Pogreska u dimenzijama slika u trening setu podataka"
assert(X_validation.shape[1:] == dimension), "Pogreska u dimenzijama slika u trening setu podataka"
assert(X_test.shape[1:] == dimension), "Pogreska u dimenzijama slika u trening setu podataka"
 
# Ucitavanje vrijednosti iz csv datoteke (oznake)
data = pd.read_csv(labelFile)

num_of_samples = []
cols = 3
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()

for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 1:
            axs[j][i].set_title(str(j), color='white')
            num_of_samples.append(len(x_selected))
 
# Ispis grafa koji pokazuje odnos broja slika i broja klasa
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Broj klase")
plt.ylabel("Broj fotografija")
plt.show()


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img


# Iterativno preprocesuiranje svih fotografija iz svih setova podataka
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# Svim setovima podataka dodijeljuje se dubina vrijednosti 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
 
# Augmentacija slika
dataGen= ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10)
dataGen.fit(X_train)

# Svakim pozivom ove funkcije generira se augmentirani set podataka veličine 20
batches= dataGen.flow(X_train,y_train,batch_size=20)
X_batch, y_batch = next(batches)
 
# Prikaz slika nakon augmentacije
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(dimension[0], dimension[1]))
    axs[i].axis('off')
plt.show()

# Funkcijom to_categorical numeričke vrijednosti oznaka pretvaraju se u vektor koristan modelu
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)


#  Kreacija modela konvolucijske neuralne mreže
def myModel():

    no_filters = 60
    filter_size = (5, 5)
    filter_size2 = (3, 3)
    pool_size = (2, 2)
    no_nodes = 500

    model = Sequential()
    model.add((Conv2D(no_filters, filter_size, input_shape=(dimension[0], dimension[1], 1), activation='relu')))
    model.add((Conv2D(no_filters, filter_size, activation='relu')))
    model.add(MaxPooling2D(pool_size=pool_size))
 
    model.add((Conv2D(no_filters//2, filter_size2, activation='relu')))
    model.add((Conv2D(no_filters // 2, filter_size2, activation='relu')))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
 
    model.add(Flatten())
    model.add(Dense(no_nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# TRENIRANJE MODELA
model = myModel()
print(model.summary())
history = model.fit(X_train, y_train, batch_size=batch_size_val, steps_per_epoch=int(steps_per_epoch_val/batch_size_val), epochs=epochs_val,validation_data=(X_validation, y_validation), shuffle=1)
# ISCRTAVANJE GRAFA VRIJEDNOSTI TRENINGA
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('training loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('training accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(X_test, y_test, batch_size=(batch_size_val*2))
print("test loss, test accuracy:", score[0])
print("test accuracy: ", str(score[1] * 100), "%")

# Model se sprema kao objekt ekstenzije .h5
try:
    model.save('model.h5')
    print('Saved')
except IOError as e:
    print('Model failed to save as a file')
    print(e)
else:
    print('Model saved successfully!')
cv2.waitKey(0)
