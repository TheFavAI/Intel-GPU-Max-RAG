# Llama2 RAG running on Intel Data Center GPU Max

Here is a python code that leverages Llama 2, LangChain, and Streamlit to deploy a RAG on Intel Data Center GPU.

It is an adaptation of the Nick Renotte RAG, see credits at the end.

You'll need Docker to run it easily as Intel containers are the best way to ensure you have all the correct dependencies.

## Installation
Here is the Pre-requisite steps to run it:
1. Install docker `sudo apt install -y docker.io`
2. Pull the Intel PyTorch docker for Intel DC GPU 
`sudo docker pull intel/intel-extension-for-pytorch:2.1.10-xpu`
3. Run the container exposing the correct ports to access Streamlit on your browse

`sudo docker run --expose 8501 -p 8501:8501 -it -u root --privileged --device=/dev/dri -v /home/:/home/ --ipc=`
4. Install the dependencies 

`pip install streamlit sentence-transformers llama-index llama_hub==0.0.19 langchain einops accelerate transformers bitsandbytes scipy`
5. Run streamlit `streamlit run App_Llama2_Intel_GPU.py`

## Credits
https://github.com/nicknochnack/Llama2RAG
