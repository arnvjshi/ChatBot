import streamlit as st
from langchain import HuggingFaceHub, LLMChain, PromptTemplate
import re
from langdetect import detect

# Streamlit UI
st.title("AI ChatBOT")

#API token
api_key = "Your API key"
# Set up the model
model = "HuggingFaceH4/starchat-beta"

# Function to create LLM and LLMChain
def create_llm_chain(api_key):
    llm = HuggingFaceHub(
        repo_id=model,
        huggingfacehub_api_token=api_key,
        model_kwargs={
            "min_length": 30,
            "max_new_tokens": 256,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 0.95,
            "eos_token_id": 49155
        }
    )
    
    prompt_template = PromptTemplate(
        input_variables=["user_input"],
        template="User: {user_input}\nChatBot: "
    )
    
    return LLMChain(prompt=prompt_template, llm=llm)

def extract_english(text):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    english_sentences = []
    for sentence in sentences:
        try:
            if detect(sentence) == 'en':
                english_sentences.append(sentence)
            else:
                # Stop when we encounter a non-English sentence
                break
        except:
            break
    
    return ' '.join(english_sentences).strip()

def generate_response(user_input):
    try:
        llm_reply = llm_chain.run(user_input)
        reply = llm_reply.split("AI:")[-1].strip()
        english_reply = extract_english(reply)
        return english_reply
    except Exception as e:
        return f"An error occurred: {str(e)}"

if api_key:
    llm_chain = create_llm_chain(api_key)
    
    user_input = st.text_input("User:", "")
    if st.button("Enter"):
        response = generate_response(user_input)
        st.text_area("ChatBot:", response, height=200)
else:
    st.warning("ERROR : Not Proper Prompt")
