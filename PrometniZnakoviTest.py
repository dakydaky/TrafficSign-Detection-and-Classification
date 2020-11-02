import numpy as np
import cv2
import os
from keras.models import load_model
import random
import string

# Postavljanje parametara za lakse promjene osjetljivih vrijednosti
threshold = 0.62  # varijabla predstavlja potrebnu vjerojatnost
font = cv2.FONT_HERSHEY_SIMPLEX
folder = 'myTests'  # put foldera gdje se nalaze testne fotografije
minSize = 30  # varijabla predstavlja najmanju veličinu koju trazi HAAR algoritam

signWidth = 60  # sirina prometnog znaka koristi se za kalkulaciju udaljenosti
focalL = 3  # fokalna duzina lece kamere koristi se za kalkulaciju udaljenosti

# Ucitavanje treniranog modela ekstenzije .h5
model = load_model('model.h5')


# Definirane funckije za predprocesuiranje ulazne slike
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


# Definirana funkcija za nasumican string (za spremanje greski)
def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


# Definirana funkcija za izracun udaljenosti kamere od trazenog prometnog znaka
def distance_to_camera(perWidth):
    # formulom trijangulacije dolazimo do aproksimacije udaljenosti
    return (signWidth * focalL) / perWidth


# Definirana funkcija za dohvacanje naziva pojedinog znaka
def get_class(index):
    labels = {0: 'Ogranicenje brzine 20 km/h',
              1: 'Ogranicenje brzine 30 km/h',
              2: 'Ogranicenje brzine 50 km/h',
              3: 'Ogranicenje brzine 60 km/h',
              4: 'Ogranicenje brzine 70 km/h',
              5: 'Ogranicenje brzine 80 km/h',
              6: 'Kraj ogranicenja brzine',
              7: 'Ogranicenje brzine 100 km/h',
              8: 'Ogranicenje brzine 120 km/h',
              9: 'Zabrana pretjecanja',
              10: 'Zabrana pretjecanja za teska vozila',
              11: 'Pravo prolaza na sljedecem raskrizju',
              12: 'Glavna cesta',
              13: 'Prilaz',
              14: 'Stop',
              15: 'Zabrana za vozila',
              16: 'Zabrana za teska vozila',
              17: 'Zabranjen prilazak',
              18: 'Oprez',
              19: 'Opasan zavoj u lijevo',
              20: 'Opasan zavoj u desno',
              21: 'Dvostruki zavoj',
              22: 'Kvrgava cesta',
              23: 'Sklizak kolnik',
              24: 'Suzenje ceste s desne strane',
              25: 'Radovi na cesti',
              26: 'Prometna signalizacija',
              27: 'Pjesacka zona',
              28: 'Prijelaz za djecu',
              29: 'Staza za bicikle',
              30: 'Oprez led/snijeg',
              31: 'Prijelaz za divlje zivotinje',
              32: 'Kraj svih ogranicenja',
              33: 'Zavoj u desno',
              34: 'Zavoj u lijevo',
              35: 'Prolaz samo ravno',
              36: 'Prolaz ravno ili u desno',
              37: 'Prolaz ravno ili u lijevo',
              38: 'Zadrzite se u desnoj traci',
              39: 'Zadrzite se u lijevoj traci',
              40: 'Obvezan kruzni tok',
              41: 'Kraj ogranicenja pretjecanja',
              42: 'Kraj ogranicenja pretjecanja za teska vozila',
              43: 'False positive',
              44: 'Zabrana zaustavljanja'
              }
    return labels[int(classIndex)]


# Program iterira kroz fotografije iz testnog foldera te se provodi prepoznavanje
for file in os.listdir(folder):
    # Ucitavanje fotografije u varijablu
    img = cv2.imread(os.path.join('myTests', file))
    # Spremanje procesiranih varijanti slike
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pokretanje HAAR kaskadnih klasifikatora i detekcija objekata
    yieldC = cv2.CascadeClassifier('cascades/yield.xml')
    allC = cv2.CascadeClassifier('cascades/all.xml')
    foundY = yieldC.detectMultiScale(img_gray, minSize=(minSize, minSize))
    allY = allC.detectMultiScale(img_gray, minSize=(minSize, minSize))
    found = []

    # Pronadjeni elementi fotografije se spremaju u listu
    for item in foundY:
        if item == []:
            continue
        found.append(item)
    for item in allY:
        if item == []:
            continue
        found.append(item)

    if len(found):
        for (x, y, width, height) in found:
            # Spremanje dijelova fotografije kao nove slike i njihovo procesuiranje
            subimg = img_rgb[y:y + height, x:x + width]
            imgNew = np.asarray(subimg)
            imgNew = cv2.resize(subimg, (32, 32))
            imgNew = preprocessing(imgNew)
            imgNew = imgNew.reshape(1, 32, 32, 1)
            # Model koristi niz kako bi predvidio vrijednosti za svaku oznaku (kategoriju)
            predictions = model.predict(imgNew)
            classIndex = model.predict_classes(imgNew)
            # Sprema se vrijednost vjerojatnosti za najizgledniju kategoriju
            probabilityValue = np.amax(predictions)
            # Ukoliko je vrijednost veca ili jednaka od zadanog praga, te ne pripada klasi koja definira greske:
            if probabilityValue > threshold and classIndex != 43:
                # Poziva se funkcija mjerenja udaljenosti
                distance = distance_to_camera(width)
                # Iscrtava se kvadrat oko znaka
                cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 5)
                # Tekstom se ispisuje kojoj klasi znak pripada, vjerojatnost te udaljenost od kamere
                cv2.putText(img,
                            str(classIndex) + " " + str(get_class(classIndex)) + ", " + str(
                                round(probabilityValue * 100, 2)) + "%, ~" + str(round(distance * 2) / 2) + " m",
                            (x - height, y), font, 0.6,
                            (0, 0, 255), 2, cv2.LINE_AA)
                # Slika se sprema u zaseban folder kako bi se model mogao ponovno trenirati na greskama
                cv2.imwrite('images/{}.png'.format(get_random_string(10)), subimg)
                print('found a match! (' + str(round(probabilityValue * 100, 2)) + ' % chance)')

    # Program prikazuje fotografije jednu za drugom, dok ne dodje do posljednje
    cv2.imshow('image', img)
    cv2.imwrite('results/{}.png'.format(get_random_string(10)), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
