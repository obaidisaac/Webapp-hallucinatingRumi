from pickle import load
from typing import List
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.responses import HTMLResponse
import tensorflow as tf
import string
from numpy import argmax
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from numpy import array


def preProcess_data(text):
    clean_text = text.split()
    table = str.maketrans('','',string.punctuation)
    clean_text = [w.translate(table) for w in clean_text]
    clean_text = [word for word in clean_text if word.isalpha()]
    clean_text = [word.lower() for word in clean_text]
    return clean_text # is a list

def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text # is a list
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = array(pad_sequences([encoded], maxlen=seq_length, padding = 'pre',truncating='pre'))
        yhat = model.predict(encoded, verbose=0)
        yhat_idx = argmax(yhat, axis=1)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat_idx:
                out_word = word
                break
        in_text += ' ' + out_word
        result.append(out_word)

    return ' '.join(result)


class ModelResponse(BaseModel):
    user_input: str
    response: str


app = FastAPI()
templates = Jinja2Templates(directory="templates/")

@app.get('/') #basic get view
async def basic_view():
    return {"WELCOME! Rumi hallucinations created by generative AI. Language trained on Rumi's poetry (Essential Rumi pub.)": "GO TO /docs route, or /post or send post request to /predict ","data":"data"}


@app.get('/predict', response_class=HTMLResponse) #data input by forms
async def take_inp():
    return '''
    <form method="post"> 
    <input type="text" maxlength="100" name="text" value="Enter a few words: "/>  
    <input type="submit"/> 
    </form>'''



@app.post('/predict', response_model=ModelResponse) #prediction on data
async def predict(text:str = Form(...)): #input is from forms
    clean_text = preProcess_data(text) #cleaning and preprocessing of the texts
    
    tokenizer = load(open('tokenizer.pkl','rb'))
    loaded_model = load_model('model.h5') #loading the saved model
    
    gen_seq = generate_seq(loaded_model, tokenizer, 74, clean_text, 50)	
    return {"user_input" : text , "response" : gen_seq}
    #return templates.TemplateResponse('predict.html', context={'request': request, 'result': gen_seq})
