from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
import sys
import os
import gradio as gr
from IPython.display import Markdown, display

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index

def ask_ai(query):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(query)
    return f"Response: *{response.response}*"

OPENAI_API_KEY="OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

construct_index("chatbotdata")

iface = gr.Interface(
    fn=ask_ai,
    inputs="text",
    outputs=gr.outputs.Textbox(),
    layout="vertical",
    title="Ask AI",
    description="Ask the AI any question.",
)

iface.launch()
