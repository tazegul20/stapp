import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings 
from langchain_community.vectorstores import FAISS
#from langchain.vectorstores import FAISS
import io
from langchain_openai.chat_models.base import ChatOpenAI
from transformers import AutoModel,AutoTokenizer
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from htmlTemplates import css,bot_template,user_template


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_file = io.BytesIO(pdf.read())
        reader=PdfReader(pdf_file)
        for page in reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks= text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')    
    vectorstore= FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectore_store):
    llm=ChatOpenAI()
    memory= ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    converstion_chain= ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectore_store.as_retriever(),
        memory=memory,
    )
    return converstion_chain
    
def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i ,message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)
            
        



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with me")

    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation= None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    
    st.header("Chat with my personal assistant :)")
    user_question=st.text_input("Ask something you wonder about me !")
    if user_question:
        handle_userinput(user_question)
     

    with st.sidebar:
        st.subheader("Document in the CV")
        pdf_docs=st.file_uploader("Upload the pdf and click on 'Process'",type="pdf",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text= get_pdf_text(pdf_docs)
                
                text_chunks = get_text_chunks(raw_text)    
                
                vectore_store= get_vectorstore(text_chunks)

                st.session_state.conversation=get_conversation_chain(vectore_store)
                


        
if __name__ == '__main__':
    main()
