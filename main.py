import os 
from openai import OpenAI
import streamlit as st
import pandas as pd
import math
import pandas as pd
import os
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer,util
from sklearn.metrics.pairwise import cosine_similarity
from dotenv.main import load_dotenv





load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def istem_atma(user_q):
    df=pd.read_excel('sorular1.xlsx')

 
    model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
    soru_list = df['Soru'].to_list()
    cevap_list = df['Cevap'].to_list()
    user_q_embedded=model.encode(user_q, convert_to_tensor=False)
    soru_embedded = model.encode(soru_list, convert_to_tensor=False)
    cevap_embedded = model.encode(cevap_list, convert_to_tensor=False)

    
    cosine_similarities=util.pytorch_cos_sim(user_q_embedded,soru_embedded)
    top_10=cosine_similarities.argsort(axis=-1)[:,-10:] # get the top 10 similar items for each
    
    for top_question in top_10:
        top_question=df['Soru'].iloc[top_question][-10:][::-1]
        
    
    ans=None
    for ans in top_10:
        ans=df['Cevap'].iloc[ans][-10:][::-1]
        
    
    answers=ans.to_list()
    
    message_text = [{"role":"system","content":f"You are an AI assistant that can only answer based on the following list: {answers[0]},{answers[1]}. Read the list  and execute the correct answer as an output. If you can't find the answer in the list just say 'Üzgünüm bu soruya cevap veremiyorum'."},
                    {"role":"user","content":f"{user_q}"}]
    
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo-16k",# model = "deployment_name"
      messages = message_text,
      temperature=0.7,
      max_tokens=800,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None,
    )
    
    a=completion.dict()
    b=a['choices'][0]['message']['content']
    return b



txt = st.text_input('Lutfen istemi giriniz')


button = st.button('Istemi gonderin')

if button:

    cevap=istem_atma(txt)
    
    
    st.write(cevap)
