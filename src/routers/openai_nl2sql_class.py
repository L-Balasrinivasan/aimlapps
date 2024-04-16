import os
from pprint import pprint

# import bitdotio
# from dotenv import load_dotenv
import typing as t

import openai
import json
import yaml
import getpass
import os
from pathlib import Path
import time
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


# Enclose with '[]' for all the column names. For Example: select [column_name_1], [column_name_2] from table.

class openai_nl2sql:

    def __init__(self):
        os.environ["OPENAI_API_KEY"] = ""
        self.llm = ChatOpenAI(temperature=0.0)

    def save_sample_prompt(self, filepath, prompt):
        with open(filepath, "w") as f:
            f.write(str(prompt))

    def load_json(self, path):
        with open(path, "r") as f:
            content = json.load(f)
        return content

    def get_json_str(self, data):
        return json.dumps(data)

    def write_json(self, filepath, data, indent=4):
        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent)

    def start_process(self, schema, query):
        system_instruction = """
            you are a text-to-SQL translator.
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. 
            Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
            If you don't know the answer to a question, please don't share false information.  
            """

        main_prompt_template = """
            Using the above {schema}, 
            Write an Azure Synapse SQL query for the following text: 
            ```{query}```
            NOTE: Don't use any sql reserved keywords as alias.
            """

        # dummy_prompt = system_instruction + main_prompt_template

        # print(dummy_prompt)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", main_prompt_template)
        ])

        schema_json_str = self.get_json_str(schema)
        chain = prompt | self.llm
        # print("prompt")
        # print(prompt)
        response = chain.invoke({"schema": schema_json_str, "query": query})

        return response.content
