import streamlit as st
import sqlite3
import pandas as pd
from openai_nl2sql_class import openai_nl2sql
from bard_class_v1 import bardflanllm
from llama2_class_v1 import llama_nl2sql
from gemini_class_v1 import Geminillm
import json

import os


# Function to execute SQL queries
def execute_query(sql_query):
    conn = sqlite3.connect('employee.db')
    cur = conn.cursor()
    try:
        cur.execute(sql_query)
        rows = cur.fetchall()
        return rows, cur.description
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None
    finally:
        cur.close()
        conn.close()

obj = None
# Streamlit app interface setup
st.title("Natural Language to SQL Query Translator and Executor")

schema_option = st.radio("Select schema:",["default","upload new schema"])

if schema_option == "default":

  # Displaying database schema images if needed
  st.image('schema01.png')
  st.image('schema02.png')
  
  with open("default_schema.json") as file:
    json_data = json.load(file)
    
  # User input for natural language query
  default_query = "Retrieve the name and contact details of all suppliers who have not provided any products yet."
  user_query = st.text_area("Enter a query description in plain English:", default_query)
  # Model selection for translating NL to SQL
  model_selection = st.selectbox("Select an LLM model", ["bard","gemini","open_ai", "flan", "llama2"])
  language_selection = st.selectbox("Select an Language", ["SQL","","KQL", "MQL"])
  # Button to submit the query for processing

else:
  uploaded_file = st.file_uploader("Upload JSON file:", type="json")

  if uploaded_file is not None:
    # Read the uploaded JSON file
    json_data = json.load(uploaded_file)

    user_query = st.text_area("Enter a query description in plain English:")

    # Model selection for translating NL to SQL
    model_selection = st.selectbox("Select an LLM model", ["bard","gemini","open_ai", "flan", "llama2"])
    
    language_selection = st.selectbox("Select an Language", ["SQL","","KQL", "MQL"])
    
    # Button to submit the query for processing
if st.button("Submit"):
  if model_selection == "open_ai":
    obj = openai_nl2sql()
  elif model_selection == "flan":
    obj = bardflanllm("flan")
  elif model_selection == "llama2":
    obj = llama_nl2sql()
  elif model_selection == "bard":
    obj = bardflanllm("bard")
  elif model_selection == "gemini":
    obj = Geminillm("gemini")
  else:
    st.write("No Model Selected")
    obj = None

if obj:
  with st.spinner("Translating and executing your query..."):
  # Translate the query
    result = obj.start_process(json_data, user_query)
            
  # Sanitize the result to ensure it only contains valid SQL query
    if result and isinstance(result, str):
  # Strip any non-SQL content such as markdown backticks
      clean_result = result.replace('```', '').strip()
      clean_result = clean_result.replace('sql', '').strip()
                
      # Display the cleaned SQL code
      st.code(clean_result, language="sql")
                
      if schema_option=="default":
        # Execute the SQL query and display results
        query_result, description = execute_query(clean_result)
        if query_result and description:
          df = pd.DataFrame(query_result, columns=[column[0] for column in description])
          st.subheader("Result:")
          st.dataframe(df)
        else:
          st.write("No data found or query error.")
