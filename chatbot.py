import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
lemmatizer = WordNetLemmatizer()
#opening the json file with the read mode 
stuffs = json.loads(open('stuffs.json').read())
words = pickle.load(open('words.pk1','rb'))
classes=pickle.load(open('classes.pk1','rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words= nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words=clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word==w:
                bag[i] = 1
    return np.array(bag)

def predict(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r> ERROR_THRESHOLD]

    results.sort(key=lambda x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'stuffs' : classes[r[0]], 'probability':str(r[1])})
    return return_list

def response(stuffs_list,stuffs_json):
    tag = stuffs_list[0]['stuffs']
    list_of_stuffs = stuffs_json['stuffs']
    for i in list_of_stuffs:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
        else:
            print("I dont know the answer for that question?")
    return result

print("GO! Bot is running")

while True :
    message = input("")
    ints = predict(message)
    res = response(ints,stuffs)
    print(res)


