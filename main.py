# ==============================================================================
# Copyright 2024 Alexander Wendt. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#
# Run with: streamlit run ./main.py -- -l

# To get it running, except of the required libraries, you also need
# conda install -c conda-forge popplery
# conda install -c conda-forge tesseract
# set #os.environ['TESSDATA_PREFIX'] = "C:/Users/alexander.wendt/.conda/envs/examextractor/share/tessdata"
#

import argparse
import logging
import os

import dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VST
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2024'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '0.1.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@gmx.at'
__status__ = 'Experimental'

parser = argparse.ArgumentParser(description='Exam question extractor')
parser.add_argument("-l", "--load_store", action='store_true', help="Load existing store", required=False)
parser.add_argument("-q", "--load_query", action='store_true', help="Load a predefined query to test with",
                    required=False)

# CONSTANTS #
STORAGE_PATH = "./database/vectorstore.pkl"
RESOURCE_PATH = "./resources"
OUTPUT_PATH = "./output"
AIMODEL = "gpt-3.5-turbo"
# EMBEDDINGSMODEL = "text-embedding-3-small" # or text-embedding-3-large, source: https://platform.openai.com/docs/guides/embeddings
EMBEDDINGSMODEL = "text-embedding-3-large"

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

log.info(args)

# Start Streamlit
# Set page configuration
st.set_page_config(
    page_title="Teacher Bot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.image("./images/teacherbotlogo.jpg", width=200)  # Add an image logo if you have one
    st.title('Model: {}'.format(AIMODEL))
    st.write("""
    Welcome to the Teacher Bot! This bot is designed to assist teachers in creating exam questions.
    Simply upload a text and the bot will either answer questions related to the text or generate a set of exam questions for you.
    """)

# Main Page
st.header("🎓 Teacher Bot 🎓")
st.subheader("Your personal assistant for creating exam questions")


def is_api_key_valid():
    """
    Check if the OpenAI key is valid

    """
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.models.list()
    except openai.AuthenticationError as e:
        log.error("Error: {}".format(e));
        return False
    else:
        return True


class ExamQuestionExtractor:
    """
    The exam question extractor main class

    """

    def __init__(self):
        """
        Init the Exam Question Extractor
        """

        log.info("Init Exam Question Extractor")
        # Load environment variables

        log.debug("Load enviroment variables")
        dotenv.load_dotenv("./.env")

        # Check open ai key
        if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
            log.error("OPENAI_API_KEY has not been set")
            exit(1)

        # log.debug("OpenAI API Key={}".format(os.getenv("OPENAI_API_KEY")))

        if not is_api_key_valid():
            log.error("Invalid API key {}".format(os.getenv("OPENAI_API_KEY")))
            exit(1)
        else:
            log.info("OpenAI key is valid")

        log.debug("Create output path if it does not exist: {}".format(OUTPUT_PATH))
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        log.debug("Create database path if it does not exist: {}".format(os.path.dirname(STORAGE_PATH)))
        os.makedirs(os.path.dirname(STORAGE_PATH), exist_ok=True)
        log.debug("Create resources path if it does not exist: {}".format(RESOURCE_PATH))
        os.makedirs(RESOURCE_PATH, exist_ok=True)

    def load_docs(self, load_storage: bool, embeddings: OpenAIEmbeddings) -> VST:
        """
        Load all docs from the defined folder or from a stored database into a vector store

        :param load_storage: load storage stored in ./database
        :param embeddings: Embeddings for the vector database
        :return:
        """

        if load_storage:
            vectorstore = FAISS.load_local(STORAGE_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            # Import documents
            loader = DirectoryLoader(RESOURCE_PATH)
            loaded_documents = loader.load()
            if len(loaded_documents) > 0:
                print("Documents: {}", loaded_documents)

                # Get from the input documents
                vectorstore = FAISS.from_documents(loaded_documents, embeddings)
                # Store database
                vectorstore.save_local(STORAGE_PATH)
            else:
                raise ImportError("No resource files were loaded")

        return vectorstore

    def define_exam_prompt(self) -> PromptTemplate:
        """
        Defines the behaviour of the chatbot.

        :return:
        """
        template = """
        You are a teacher for a class in an Austrian Gymnasium. You love to teach students and to help them. 
        Students in the age of 12-15 will ask you one of two things: 
        - Case 1: to create a number of exam questions from the given text, as well as answers to those questions
        - Case 2: or to answer a specific question by a student about the exam topic.
        
        Your first task is to read the question of the student: {question}
        
        Then you shall decide if the question is related to case 1 or case 2.
        
        Use the context is provided in the following within the ```CONTEXT``` to fulfill the task.

        Context: ```{context}```
              
        In case 1, the student is expecting question and answer pairs to be returned in the following format:
        Question: 
        Answer:
        
        If more than one question-answer pair are generated, leave a blank line between the questions. Do not return 
        anything else, but the Question-Answer pairs.
        
        In case 2, the student is expecting to get both his or her question returned and an answer in the following format:        
        Question: 
        Answer: 
    
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        return prompt

    def format_docs(self, docs) -> str:
        """
        Adapt doc format to return the docs as a string

        :param docs: docs list
        :return: docs string
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def execute_program(self, load_storage: bool, load_query: bool) -> None:
        """
        Execute the program parts

        :param load_storage: load storage stored in ./database
        :param load_query: load a predefined query for debugging purposes
        """

        # Load environment variables. They are used by openai
        embeddings = OpenAIEmbeddings(model=EMBEDDINGSMODEL)

        # Get docs into store
        vectorstore = self.load_docs(load_storage, embeddings)

        # Generate question prompt
        prompt = self.define_exam_prompt()

        # Load model
        llm = ChatOpenAI(temperature=0, model_name=AIMODEL)

        query = st.text_input('Enter a question or the number of questions to be generated:')

        if load_query or query:
            if (load_query):
                # query = "Generate me 5 questions about the Reading text of unit 8?"
                # query = "How long did it take until her idea was found by the military?"
                # query = "What does cheap mean in German?"
                # query = "From the Words and Phrases, generate 5 words to translate"
                query = "Generate 3 exam questions about Hedy Lamarr based on the provided text"

            # getting only the chunks that are similar to the query for llm to produce the output
            similar_embeddings_query = vectorstore.similarity_search(query)
            similar_embeddings = FAISS.from_documents(documents=similar_embeddings_query,
                                                      embedding=embeddings)

            # creating the chain for integrating llm,prompt,stroutputparser
            retriever = similar_embeddings.as_retriever()
            rag_chain = (
                    {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
            )

            response = rag_chain.invoke(query)

            log.debug("AI response: \n{}".format(response))
            st.write(response)


if __name__ == "__main__":
    """
    I am your exam question generator.
    """
    # streamlit run ./main.py -- -l

    log.info("=== Start Exam Question Extractor ===")

    exam = ExamQuestionExtractor()
    exam.execute_program(args.load_store, args.load_query)

    log.info("=== Program end ===")
