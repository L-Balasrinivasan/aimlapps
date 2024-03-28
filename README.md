# Read websites using LangChain

## How to run?

1. Run the init setup file first in any bash terminal

    ```bash
    bash init_setup.sh
    ```

2. Activate the conda environment

    ```bash
    conda activate ./env
    ```

3. run the main app using -

    ```bash
    python main.py
    ```

    or

    ```bash
    uvicorn main:app --reload
    ```

## How it works?

![workflow-diagram](./images/qa_flow-9fbd91de9282eb806bda1c6db501ecec.jpeg)

Summary of the steps given in the diagram to do QnA on a document or text obtained from a URL:

1. **Document Loading**: The document or text is obtained from the URL and loaded into the system.
2. **Splitting**: The document is split into smaller parts or “splits” for easier processing.
3. **Storage**: The splits are stored in a vector store for later retrieval.
4. **Retrieval**: The relevant splits are retrieved from the vector store based on the user’s query or question.
5. **Output**: The relevant splits are processed by a language model (LM) to generate an answer for the user.

This process allows for efficient and accurate Q&A on large documents or texts.

---

This process involves creating a Q&A system that uses AI to handle large documents or texts. It leverages an existing AI model, which has been trained on a vast amount of data. This model is capable of understanding and categorizing the content of the document. The categorized content is then used to answer queries related to the document.

**Business Value:**

- **Efficiency**: It saves time and resources as you don’t need to train a complex AI model from scratch.
- **Scalability**: It can handle large documents and high volumes of queries, making it scalable for growing businesses.
- **Accuracy**: The use of an AI language model ensures accurate and contextually relevant answers.

**Applications:**

- **Customer Support**: Automating FAQ or helpdesk services, reducing response time, and improving customer satisfaction.
- **Research & Development**: Quick access to relevant information from large research papers or reports.
- **Legal & Compliance**: Easy retrieval of specific clauses or sections from large legal documents or contracts.
- **Education**: Assisting students in finding relevant information from textbooks or lecture notes.

This approach provides an efficient, scalable, and accurate solution for businesses looking to automate their Q&A services on large documents or texts.
