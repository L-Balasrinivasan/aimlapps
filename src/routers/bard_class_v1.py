# import google.generativeai as palm
from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
import typing as t
import yaml
import json
import os
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import json
from pathlib import Path
from langchain.llms import GooglePalm
import google.generativeai as genai
import time
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


class bardflanllm:

    def __init__(self, llm_model):
        if llm_model == "bard":
            print("selected Bard as the LLM Model")
            os.environ["GOOGLE_API_KEY"] = "AIzaSyA2FmPI7w_LnhIE1Q2qr0W7n9qBCCzSlXM"
            self.llm = GooglePalm(temperature=0)
        
        if llm_model == "flan":
            print("selected Flan as the LLM Model")
            os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
            self.llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                                      model_kwargs={"temperature": 0.001, "max_length": 1000000})

    def parse_json_schema(self, json_data):
        formatted_json_string = json.dumps(json_data, indent=2)
        return formatted_json_string

    def start_process(self, json_data, query, query_langauge):

        system_instruction = """
            you are a text-to-{query_langauge} translator.
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. 
            Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
            If you don't know the answer to a question, please don't share false information.  
        """
        main_prompt_template = """
            Using the below schema: {schema}, 
            Write an {query_langauge} server query for the following text: 
            ```{query}```
            NOTE: Pay attention to the syntax to generate error free queries for example do not use any {query_langauge} reserved keywords as alias.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", main_prompt_template)
        ])

        print("*" * 100)
        print("prompt")
        print(prompt)
        print("*" * 100)

        schema_json_str = self.parse_json_schema(json_data)
        chain = prompt | self.llm
        response = chain.invoke({"schema": schema_json_str, "query": query, "query_langauge":query_langauge})

        # time.sleep(1)
        # try:
        #     chain = prompt | self.llm
        #     response = chain.invoke({"schema": schema_json_str, "query": query})

        #     # with open("sameer_result.txt", "w") as f:
        #     #     f.write(str(response.content))
        # except Exception as exe:
        #     raise exe
        # finally:
        #     print("DONE")

        return response
