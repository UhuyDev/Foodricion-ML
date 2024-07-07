import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
import json
import pickle

# Load trained data
words = pickle.load(open('../chatbotmodel/words.pkl', 'rb'))
classes = pickle.load(open('../chatbotmodel/classes.pkl', 'rb'))
model = tf.keras.models.load_model('../chatbotmodel/chatbot.h5')

def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

p = bow("vitamin c", words)
res = model.predict(np.array([p]))[0]
ERROR_THRESHOLD = 0.25
results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
results.sort(key=lambda x: x[1], reverse=True)
return_list = []
for r in results:
    return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
print(return_list)