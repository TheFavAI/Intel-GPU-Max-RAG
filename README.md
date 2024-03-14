# Llama2 RAG running on Intel Data Center GPU Max

Here is a python code that leverages Llama 2, LangChain, and Streamlit to deploy a RAG on Intel Data Center GPU.
You'll need Docker to run it easily as Intel containers are the best way to ensure you have all the correct dependencies.

## Installation
Here is the Pre-requisite steps to run it:
```bash
sudo apt install -y docker.io
```
```bash
sudo docker pull intel/intel-extension-for-pytorch:2.1.10-xpu
```
```bash
pip install streamlit sentence-transformers llama-index llama_hub==0.0.19 langchain einops accelerate transformers bitsandbytes scipy
```
```bash
streamlit run App_Llama2_Intel_GPU.py
```

## Credits
https://github.com/nicknochnack/Llama2RAG
