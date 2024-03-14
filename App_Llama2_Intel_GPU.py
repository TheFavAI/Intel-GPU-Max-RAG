# Import
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import intel_extension_for_pytorch as ipex
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, download_loader
from pathlib import Path

# Define variable to hold llama2 weights naming
name = "meta-llama/Llama-2-7b-chat-hf"
# Auth token variable from hugging face
auth_token = "YOUR TOKEN HERE"

@st.cache_resource
def get_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/'
                            , use_auth_token=auth_token, torch_dtype=torch.float16,
                            rope_scaling={"type": "dynamic", "factor": 2}).to("xpu")
    model = ipex.optimize(model)

    return tokenizer, model
tokenizer, model = get_tokenizer_model()

# Create a system prompt
system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain
why instead of answering something not correct. If you don't know the answer
to a question, please don't share false information.

Your goal is to provide answers relating to the financial performance of
the company.<</SYS>>
"""
# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)

embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)

set_global_service_context(service_context)

# PDF 
PyMuPDFReader = download_loader("PyMuPDFReader")
loader = PyMuPDFReader()
file_path_str = str(Path('./data/annualreport.pdf'))
documents = loader.load(file_path=file_path_str, metadata=True)

# Create an index - we'll be able to query this in a sec
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Create centered main title
st.title('🦙 Internal Llama OpenStack resources')
# Create a text input box for the user
prompt = st.text_input('Your question :')

# If the user hits enter

if prompt:
    response = query_engine.query(prompt)
    print(response)
    # ...and write it out to the screen
    st.write(response)
