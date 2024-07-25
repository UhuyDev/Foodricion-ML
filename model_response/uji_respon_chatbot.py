import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
import json
import pickle
from transformers import TFBertModel, BertTokenizer

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("cahya/bert-base-indonesian-522M")
bert_model = TFBertModel.from_pretrained("cahya/bert-base-indonesian-522M")

# Load trained data
words = pickle.load(open('../chatbotmodel/words.pkl', 'rb'))
classes = pickle.load(open('../chatbotmodel/classes.pkl', 'rb'))
model = tf.keras.models.load_model('../chatbotmodel/chatbot.h5')

# Load intents file
with open('../foodricionchatbotdataset/intents.json') as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True)
    outputs = bert_model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    mean_embeddings = tf.reduce_mean(last_hidden_state, axis=1)
    return mean_embeddings.numpy()

def predict_class(sentence):
    # Get BERT embeddings for the input sentence
    embeddings = get_bert_embeddings([sentence])
    # Reshape embeddings to match model input shape (batch_size, time_steps, features)
    embeddings = np.expand_dims(embeddings, axis=1)  # Shape (1, 1, 768)

    # Make prediction
    res = model.predict(embeddings)[0]

    # Process the results
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])
    return "Sorry, I didn't understand that."

# Main function to handle user input
def chatbot_response(msg):
    ints = predict_class(msg)
    res = get_response(ints, intents)
    return res

# Example usage
if __name__ == "__main__":
    sentence = "apa saja komponen diet seimbang"
    response = chatbot_response(sentence)
    print(response)
