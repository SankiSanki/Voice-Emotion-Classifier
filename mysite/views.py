from django.http import HttpResponse
from django.shortcuts import render
import librosa
import joblib
import numpy as np
import pandas as pd
import pickle
import keras
import tensorflow
import warnings

warnings.filterwarnings('ignore')

def home(request):
    return render(request,'home.html')

def result(request):
        
    path = request.FILES['audiofile']
    reslist = pred(path)
    res = reslist[0]
    emo = res.split('_')[1].capitalize()
    return render(request,'result.html',{ "emo" : emo })


def pred(file):
    model = joblib.load('final_model.sav')
    X, sample_rate = librosa.load(file
                              ,duration=2.5
                              ,sr=44100
                              ,offset=0.5
                             )

    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    newdf = pd.DataFrame(data=mfccs).T
    
    for i in range(len(newdf.columns),216):
        newdf[f'{i}'] = 0
        
    newdf= np.expand_dims(newdf, axis=2)
    newpred = model.predict(newdf, 
                         batch_size=16, 
                         verbose=1)

    filename = 'labels'
    infile = open(filename,'rb')
    lb = pickle.load(infile)
    infile.close()

    final = newpred.argmax(axis=1)
    final = final.astype(int).flatten()
    final = (lb.inverse_transform((final)))
    return final