from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
from dotenv import load_dotenv
import typing as t
import yaml
import json
import os

from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import time

class llama_nl2sql:

    def __init__(self):

        # self.llm = CTransformers(model='models_k/llama-2-7b-chat.ggmlv3.q2_K.bin',
        #                          model_type='llama',
        #                          config={'max_new_tokens': 128,
        #                                  'temperature': 0}
        #                          )
        # large quant
        self.llm = CTransformers(model='models_k/llama-2-7b.ggmlv3.q8_0.bin',
                         model_type='llama',
                         config={'max_new_tokens': 128,
                                 'temperature': 0}
                         )

    def parse_json_schema(self, json_data):
        formatted_json_string = json.dumps(json_data, indent=2)
        return formatted_json_string


    def start_process(self, json_data, query):

        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


        system_instruction = """
            you are a text-to-SQL translator.
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. 
            Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
            If you don't know the answer to a question, please don't share false information.  
        """
        main_prompt_template = """
            Using the below schema: {schema}, 
            Write an Azure Synapse SQL query for the following text: 
            ```{query}```
            NOTE: Pay attention to the syntax to generate error free queries for example do not use any SQL reserved keywords as alias.
        """

        SYSTEM_PROMPT = B_SYS + system_instruction + E_SYS
        template = B_INST + SYSTEM_PROMPT + main_prompt_template + E_INST
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", template)
        ])
        
        print("*"*100)
        print("prompt")
        print(prompt)
        print("*"*100)


        query = "Detect Players with Sudden Increases in Transaction Amounts. "
        schema_json_str  = self.parse_json_schema(json_data)

        time.sleep(1)
        try:
            chain = prompt | self.llm
            response = chain.invoke({"schema": schema_json_str, "query": query})

            # with open("sameer_result.txt", "w") as f:
            #     f.write(str(response.content))
        except Exception as exe:
            raise exe
        finally:
            print("DONE")

        return response

    # def start_process(self, json_data, query):
    #     prompt_strings = self.parse_schema(json_data)
    #     prompt_strings_str = ' '.join([str(elem) for elem in prompt_strings])

    #     # Call concat_prompt function with the list of strings
    #     result_prompt = self.get_default_prompt(query, prompt_strings_str)

    #     B_INST, E_INST = "[INST]", "[/INST]"
    #     B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    #     DEFAULT_SYSTEM_PROMPT = """\
    #     you are a text-to-SQL translator.
    #     You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. 
    #     Please ensure that your responses are socially unbiased and positive in nature.

    #     If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
    #     If you don't know the answer to a question, please don't share false information."""

    #     # instruction = "Convert the following text from English to French: \n\n {text}"
    #     # instruction = result_prompt + " Based on the database schema provided to you \n Convert the following text to sql query: \n\n {text} \n only display the sql query"
    #     instruction = result_prompt + "{text} "


    #     SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

    #     template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    #     print(template)

    #     prompt = PromptTemplate(template=template, input_variables=["text"])

    #     LLM_Chain = LLMChain(prompt=prompt, llm=self.llm)
    #     result = LLM_Chain.run(query)

    #     print('*'* 100)
    #     print("result: ")
    #     print(result)

    #     matches = re.search(r'SELECT(.*?);', result, re.DOTALL)

    #     if matches:
    #         result_query = matches.group(1).strip()
    #         result_query = ' '.join(result_query.split())  # Remove extra spaces
    #         result_query = "SELECT " + result_query
    #         # print('SELECT ' + result_query)
    #     else:
    #         print("No match found.")
    #         pattern = r"```sql(.*?)```"
    #         result_query = re.findall(pattern, result, re.DOTALL)

    #     return result_query
