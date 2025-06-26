import streamlit as st
import joblib
import re, string
def clean_text(text):
  text=text.lower()
  text=re.sub(r"i'm","i am",text)
  text=re.sub(r"he's","he is",text)
  text=re.sub(r"she's","she is",text)
  text=re.sub(r"that's","that is",text)
  text=re.sub(r"what's","what is",text)
  text=re.sub(r"where's","where is",text)
  text=re.sub(r"\'ll"," will",text)
  text=re.sub(r"\'ve"," have",text)
  text=re.sub(r"\'re"," are",text)
  text=re.sub(r"\'d"," would",text)
  text=re.sub(r"won't","will not",text)
  text=re.sub(r"can't","can not",text)
  text=re.sub(r"don't","do not",text)
  text=re.sub(r"shouldn't","should not",text)
  text=re.sub(r"wouldn't","would not",text)
  text=re.sub(r"couldn't","could not",text)
  text=re.sub(r"\+d",'',text)
  text=re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]",'',text)
  text=text.translate(str.maketrans('','',string.punctuation))
  text=re.sub(r"\s+"," ",text)
  return text
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("spam_model.pkl")
st.title("Spam Detection(Email/SMS)")
input_sms=st.text_area("Enter Your Email or SMS:",height=300)
if st.button("Predict"):
  if input_sms.strip() == "":
        st.warning("Please enter some text.")
  else:
    cleaned=clean_text(input_sms)
    vector=tfidf.transform([cleaned])
    prob = model.predict_proba(vector)[0][1]
    prediction = (prob > 0.29).astype(int) 
    if prediction ==0:
      st.success("Not Spam")
    else:
      st.error("Spam")

