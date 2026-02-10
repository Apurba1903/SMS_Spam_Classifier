import nltk
import string
import pickle
import streamlit as st
from  nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



def transform_text(text):
    
    # Lower Case
    text = text.lower()
    
    # Tokenization
    text = nltk.word_tokenize(text)
    
    # Removing Special Characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    # Removing Stop Words and Punctuations
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    # Stemming
    ps = PorterStemmer()
    
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i)) 
    
    
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("📩 Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the Message")

if st.button('Predict'):

    # 1. PreProcess
    transform_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transform_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("🚨 Spam")
    else:
        st.header("✅ Not Spam")