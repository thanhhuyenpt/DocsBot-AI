import os
from dotenv import load_dotenv
load_dotenv()

import openai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import (StuffDocumentsChain, 
                              LLMChain)
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.manager import trace_as_chain_group

import gradio as gr

from constants import CHROMA_SETTINGS

"""
import LANGCHAIN_API_KEY in case you encounter the error: 
langsmith.utils.LangSmithUserError: API key must be provided when using hosted LangSmith API
"""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__80c61538efab4b67a9cdf63022e62cfa"


embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# Create and store locally vectorstore
db = Chroma(embedding_function=embeddings,
            persist_directory=persist_directory,
            client_settings=CHROMA_SETTINGS)

# Set up our retriever
retriever = db.as_retriever()

# Define llm
llm = ChatOpenAI(temperature=0, openai_api_key='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')

"""
Set up our chain that can answer questions based on documents:
This controls how each document will be formatted. Specifically,
it will be passed to `format_document` - see that function for more details
"""
document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
)
document_variable_name = "context"
# The prompt here should take as an input variable the `document_variable_name`
prompt_template = """Use the following pieces of context to answer user questions. 
If you don't know the answer, just say that can not found in knowledge base, 
don't try to make up an answer.

--------------

{context}"""
system_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
prompt = ChatPromptTemplate(
    messages=[
        system_prompt, 
        MessagesPlaceholder(variable_name="chat_history"), 
        HumanMessagePromptTemplate.from_template("{question}")
	]
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
combine_docs_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
    document_separator="---------"
)

"""
Set up a chain that controls how the search query for the vectorstore is generated:
This controls how the search query is generated.
Should take `chat_history` and `question` as input variables.
"""
template = """Combine the chat history and follow up question into a a search query.

Chat History:

{chat_history}

Follow up question: {question}
"""
prompt = PromptTemplate.from_template(template)
question_generator_chain = LLMChain(llm=llm, prompt=prompt)

# Function to use
def qa_response(message, history):
	# Convert message history into format for the `question_generator_chain`.
	convo_string = "\n\n".join([f"Human: {h}\nAssistant: {a}" for h, a in history])

	# Convert message history into LangChain format for the final response chain.
	messages = []
	for human, ai in history:
		messages.append(HumanMessage(content=human))
		messages.append(AIMessage(content=ai))

	# Wrap all actual calls to chains in a trace group.
	with trace_as_chain_group("qa_response") as group_manager:
		# Generate search query.
		search_query = question_generator_chain.run(
			question=message, 
			chat_history=convo_string, 
			callbacks=group_manager
		)

		# Retrieve relevant docs.
		docs = retriever.get_relevant_documents(search_query, callbacks=group_manager)

		# Answer question.
		return combine_docs_chain.run(
			input_documents=docs, 
			chat_history=messages, 
			question=message, 
			callbacks=group_manager
		)
	
# start the app
gr.ChatInterface(qa_response).launch(share=True, debug=True)
