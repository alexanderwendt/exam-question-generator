# Exam Question Generator

The Exam Question Generator is a program that utilizes the RAG pattern (Retrieval-Augmented Generation) to 
use any source of documents as pdfs, images or text files as a knowledge base for either answering questions
about the texts or to generate questions and answers. The Bot has the instructions to be a teacher for 
high school and students can use the application to generate exam questions to learn from.

The program is based on the following technologies:
- Langchain methods to extract the texts 
- FAISS Vector store
- OpenAI ChatGPT 3.5 Turbo
- User interface in Streamlit

## Setup

Create a python environment with the following modules
```
conda create --name examextractor python=3.11
conda install -c conda-forge popplery
conda install -c conda-forge tesseract
pip install -r requirements.txt

```
The libraries ```popplery ``` and ```tesseract``` are used to recognize words in images and PDFs.

Rename ```env.example``` to ````.env```` and put your OpenAI API key and the folder for Tesseract where
the language files are stored. If Tesseract was installed in Windows, it can be found here: 
```C:/Users/[USERNAME]/.conda/envs/examextractor/share/tessdata```

## How to run
Put your documents into the ```./resources``` folder. These documents will be your exam base

Start the program. The program has a ```main.py``` with two parameters:
- -l: if set, it loads the previously stored database instead of creating a new database from the documents in the resource folder
- -q: used when debugging to set an initial query that will trigger an answer from the chatbot

To run, type the following
```streamlit run ./main.py```

The database will be recreated with the documents in ```./resources```

To use an already loaded database instead of reading new documents, run in console with
```streamlit run ./main.py -- -l```

## Models
For the FAISS vector store per default, the model ```EMBEDDINGSMODEL = "text-embedding-3-large"``` is set

For the chat bot, the model ```AIMODEL = "gpt-3.5-turbo"``` is used
## Sources
- https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf/
- https://medium.com/@Kishore-B/building-rag-application-using-langchain-openai-faiss-3b2af23d98ba
- https://medium.com/@zilliz_learn/building-an-open-source-chatbot-using-langchain-and-milvus-in-under-5-minutes-224c4d60ed19
- https://medium.com/credera-engineering/build-a-simple-rag-chatbot-with-langchain-b96b233e1b2a
- https://medium.com/@gokulprasath100702/exploring-image-to-text-to-story-an-excursion-with-langchain-0546a067dcb1