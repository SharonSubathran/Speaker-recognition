
get_ipython().magic('pylab inline')
import os
import pandas as pd
import librosa
from librosa import display
from sklearn.preprocessing import LabelEncoder
import glob 

data, sampling_rate = librosa.load('/Users/sharonsubathransubathran/Desktop/common_voice_corpus/cv-other-dev/sample-000000.mp3')
m= librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)




pd.DataFrame(m).T.plot(figsize=(70,40))


def mfccprep(mfccs):
    mfcc = {}
    for idx, val in enumerate(mfccs):
        mfcc['mfcc'+str(idx+1)]=val
    return mfcc

#Find all filenames in the folder
l = [x[2] for x in os.walk('/Users/sharonsubathran/Desktop/common_voice_corpus/cv-other-dev')]
l=l[0]



rows = []
feature = {}

#Load the csv into a dataframe and shows the relationship between audio clip and the features like gender and ethinicty of the speaker

csv = pd.read_csv('/Users/sharonsubathran/Desktop/common_voice_corpus/cv-other-dev.csv')
csv = csv[['filename','text','age', 'gender']].dropna()

#for every file in folder-
for x in l:
    print x
    file_name = '/Users/sharonsubathran/Desktop/common_voice_corpus/cv-other-dev/'+str(x)
    print file_name
    #load the mp3 file in this path and retrieves X is audio time series and its sample rate
    X, sample_rate = librosa.load(file_name)
    #retieves mfccs and finds the mean across the 13 mfccs separately
    mfccs = list(pd.DataFrame(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)).T.mean())
    feature = mfccprep(mfccs)
    f ='cv-other-dev/'+x
    try:
        feature['age'] = list(csv[csv.filename==f].age)[0]
    except:
        feature['age']= None
    try:
        feature['gender'] = list(csv[csv.filename==f].gender)[0]
    except:
        feature['gender']=None
    rows.append(feature)


#storing all data retrieved into a dataframe
df = pd.DataFrame.from_dict(rows)
df = df.dropna()
df['gender'] = df.gender.apply(lambda x: 1 if x=='male' else 0)
agekeys = {'thirties':3, 'twenties':2, 'sixties':6, 'fourties':4, 'fifties':5, 'teens':1,
       'seventies':7, 'eighties':8}
df.age = df.age.apply(lambda x: agekeys[x])


# df.to_csv('/Users/sharonsubathran/Desktop/common_voice_corpus/data.csv',  index=Fals)
df= pd.read_csv('/Users/sharonsubathran/Desktop/common_voice_corpus/data.csv')
df.head()



import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics


X = df.drop('age', 1)
y = df.age
lb = LabelEncoder()

#converts labels into categorical data
y = np_utils.to_categorical(lb.fit_transform(y))

#splits training and test data 75/25
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


num_labels = y.shape[1]
filter_size = 2

# build neural network model
model = Sequential()

model.add(Dense(256, input_shape=(14,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

#fits the model and validates output with test data.
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

