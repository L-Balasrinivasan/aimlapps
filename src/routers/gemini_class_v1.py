import json
import os

import json
import google.generativeai as genai

class Geminillm:

    def __init__(self, llm_model):
        if llm_model == "gemini":
            print("selected Gemini as the LLM Model")
            os.environ["GOOGLE_API_KEY"] = "AIzaSyA2FmPI7w_LnhIE1Q2qr0W7n9qBCCzSlXM"
            self.llm=genai.GenerativeModel('gemini-pro')
            
    def parse_json_schema(self, json_data):
        formatted_json_string = json.dumps(json_data, indent=2)
        return formatted_json_string


    def start_process(self, json_data, query):


        prompt = """
            you are a text-to-SQL translator.
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. 
            Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
            If you don't know the answer to a question, please don't share false information.  

            Using the below schema: 
            {schema}, 
            Write an SQL query for the following text: 
            ```{query}```
            NOTE: Pay attention to the syntax to generate error free queries for example do not use any SQL reserved keywords as alias.
        """

        schema_json_str  = self.parse_json_schema(json_data)
        response=self.llm.generate_content([prompt[0],query,schema_json_str])
        return response.text
        
