from os import system
import subprocess
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain import SQLDatabase

from pathlib import Path


import streamlit as st

st.set_page_config(
    page_title="Google Langchain Agents", page_icon="🦜", layout="wide", initial_sidebar_state="collapsed"
)


ACTOR_PREFIX             = "ggl"
VERSION                  = 'v1'

PROJECT_ID               = 'cpg-cdp'
BUCKET_NAME              = f'zghost-{ACTOR_PREFIX}-{VERSION}-{PROJECT_ID}'
BUCKET_URI               = f'gs://{BUCKET_NAME}'



# system(f'gsutil cp {BUCKET_URI}/config/notebook_env.py local_config.txt')

# with open('local_config.txt') as cfg_f:
#     _ = [(exec(line), print(line)) for line in cfg_f]


# cfg_f.close()

PROJECT_ID               = "cpg-cdp"

PROJECT_NUM              = "939655404703"

LOCATION                 = "us-central1"



REGION                   = "us-central1"

BQ_LOCATION              = "US"

VPC_NETWORK_NAME         = "genai-haystack-vpc"



CREATE_NEW_ASSETS        = "True"

ACTOR_PREFIX             = "ggl"

VERSION                  = "v1"

ACTOR_NAME               = "google"

ACTOR_CATEGORY           = "technology"



BUCKET_NAME              = "zghost-ggl-v1-cpg-cdp"

EMBEDDING_DIR_BUCKET     = "zghost-ggl-v1-cpg-cdp-emd-dir"



BUCKET_URI               = "gs://zghost-ggl-v1-cpg-cdp"

EMBEDDING_DIR_BUCKET_URI = "gs://zghost-ggl-v1-cpg-cdp-emd-dir"



VPC_NETWORK_FULL         = "projects/939655404703/global/networks/me-network"



ME_INDEX_NAME            = "vectorstore_ggl_v1"

ME_INDEX_ENDPOINT_NAME   = "vectorstore_ggl_v1_endpoint"

ME_DIMENSIONS            = "768"



MY_BQ_DATASET            = "zghost_ggl_v1"

MY_BQ_TRENDS_DATASET     = "zghost_ggl_v1_trends"


import sys
import os
sys.path.append("..")

from zeitghost.agents.LangchainAgent import LangchainAgent
from zeitghost.vertex.LLM import VertexLLM
from langchain.llms import VertexAI
from zeitghost.vertex.Embeddings import VertexEmbeddings

from zeitghost.capturing_callback_handler import playback_callbacks

from zeitghost.vertex.MatchingEngineCRUD import MatchingEngineCRUD
from zeitghost.vertex.MatchingEngineVectorstore import MatchingEngineVectorStore

from zeitghost.agents.Helpers import MyCustomHandler

# Google Cloud
from google.cloud import aiplatform as vertex_ai
from google.cloud import storage
from google.cloud import bigquery
from google.cloud.aiplatform_v1 import IndexServiceClient, IndexEndpointServiceClient
from google.cloud import secretmanager

# langchain
from langchain.document_loaders import DataFrameLoader
from langchain.docstore.document import Document
from langchain import PromptTemplate

from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents.tools import Tool
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.manager import CallbackManager
from langchain import LLMMathChain, SQLDatabaseChain

from langchain.sql_database import SQLDatabase
from langchain.utilities import GoogleSearchAPIWrapper

import pandas as pd
import uuid
import numpy as np
import json
import time
import io

from pydantic import BaseModel, Field

from PIL import Image, ImageDraw
import logging
logging.basicConfig(level = logging.INFO)

#Instantiate Google cloud SDK clients
storage_client = storage.Client(project=PROJECT_ID)

vertex_ai.init(project=PROJECT_ID,location=LOCATION)

# bigquery client
bqclient = bigquery.Client(
    project=PROJECT_ID,
    # location=LOCATION
)

## Instantiate the Vertex AI resources, Agents, and Tools
# mengine = MatchingEngineCRUD(
#     project_id=PROJECT_ID 
#     , project_num=PROJECT_NUM
#     , region=LOCATION 
#     , index_name=ME_INDEX_NAME
#     , vpc_network_name=VPC_NETWORK_FULL
# )

# ME_INDEX_RESOURCE_NAME, ME_INDEX_ENDPOINT_ID = mengine.get_index_and_endpoint()
# ME_INDEX_ID=ME_INDEX_RESOURCE_NAME.split("/")[5]


# REQUESTS_PER_MINUTE = 200 # project quota==300
# vertex_embedding = VertexEmbeddings(requests_per_minute=REQUESTS_PER_MINUTE)

# me = MatchingEngineVectorStore.from_components(
#     project_id=PROJECT_ID
#     , region=LOCATION
#     , gcs_bucket_name=BUCKET_NAME
#     , embedding=vertex_embedding
#     , index_id=ME_INDEX_ID
#     , endpoint_id=ME_INDEX_ENDPOINT_ID
#     , k = 10
# )

### Create VectorStore Agent tool

# vertex_langchain_agent = LangchainAgent()

# agent_executor = vertex_langchain_agent.get_vectorstore_agent(
#     vectorstore=me
#     , vectorstore_name=f"news on {ACTOR_NAME}"
#     , vectorstore_description=f"a vectorstore containing news articles and current events for {ACTOR_NAME}."
# )

# me_tools = agent_executor.tools

## BigQuery Agent 

# TRENDS_DATASET  = MY_BQ_TRENDS_DATASET
# TRENDS_TABLE_ID = 'top_rising_terms'

# vertex_langchain_agent = LangchainAgent()
# bq_agent = vertex_langchain_agent.get_bigquery_agent(PROJECT_ID)


# bq_agent_tools = bq_agent.tools

# bq_agent_tools[0].description = bq_agent_tools[0].description + \
#  f"""
#   only use the schema {TRENDS_DATASET}
#   NOTE YOU CANNOT DO OPERATIONS AN AN AGGREGATED FIELD UNLESS IT IS IN A CTE WHICH IS ALLOWED
#   also - use a like operator for the term field e.g. WHERE term LIKE '%keyword%' 
#   make sure to lower case the term in the WHERE clause
#   be sure to LIMIT 100 for all queries
#   if you don't have a LIMIT 100, there will be problems
#  """

os.environ["GOOGLE_CSE_ID"] = 'c50dd815ac39c421f'
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDZwaalC3yz5rcgHovLCdv4wB8JlJGB_28'



## Build an Agent that has access to Multiple Tools

llm = VertexAI(
    temperature=0
    , max_output_tokens=1000
    , top_p=0.7
    , top_k=40
)

dataset = 'google_trends_my_project'

db = SQLDatabase.from_uri(f"bigquery://{PROJECT_ID}/{dataset}")

db_chain = SQLDatabaseChain.from_llm(llm, db)

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

search = GoogleSearchAPIWrapper() #IF YOU DON'T PERIODICALLY REFRESH THIS YOU WILL NOT GET RESULTS

class CalculatorInput(BaseModel):
    question: str = Field()
    
tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
        args_schema = CalculatorInput
    ),
    Tool(
        name = "Google Search",
        description="Search Google for recent results.",
        func=search.run,
    ),
    Tool(
        name="Bigquery Google Trends",
        func=db_chain.run,
        description="useful for when you need to answer questions about Google Trends. Input should be in the form of a question containing full context",
    ),
]

# Apply the correct args schema to the tools

# class BaseSchema(BaseModel):
#     # action: str = Field()
#     action_input: str = Field()

# new_tools_with_args_schema = []

# for tool in bq_agent_tools:
#     tool.args_schema = BaseSchema
    # new_tools_with_args_schema.append(tool)
    
# tools.extend(new_tools_with_args_schema)


# Get the agent

# def get_plan_and_execute_agent(llm=llm, tools=tools):
#     model = llm
#     planner = load_chat_planner(model)
#     executor = load_agent_executor(model, tools, verbose=True)
#     agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
#     return agent

# agent = get_plan_and_execute_agent(llm=llm, tools=tools)

# Run the streamlit app

SAVED_SESSIONS = {
    "how does google's commitment to technology to change the world?": "Google is committed to using technology to change the world for the better. They are working on a number of projects that aim to improve people's lives, including sustainability plans, environmental initiatives, and philanthropy programs. They are also committed to helping expand learning for everyone.",
    "What does google do to help customers with their supply chains?": "Google provides multiple solutions that help businesses build their supply chains with data and AI.",
}

"# 🦜🔗 Vertex-Palm Agents"
#helper classes
class DirtyState:
    NOT_DIRTY = "NOT_DIRTY"
    DIRTY = "DIRTY"
    UNHANDLED_SUBMIT = "UNHANDLED_SUBMIT"


def get_dirty_state() -> str:
    return st.session_state.get("dirty_state", DirtyState.NOT_DIRTY)


def set_dirty_state(state: str) -> None:
    st.session_state["dirty_state"] = state


def with_clear_container(submit_clicked: bool) -> bool:
    if get_dirty_state() == DirtyState.DIRTY:
        if submit_clicked:
            set_dirty_state(DirtyState.UNHANDLED_SUBMIT)
            st.experimental_rerun()
        else:
            set_dirty_state(DirtyState.NOT_DIRTY)

    if submit_clicked or get_dirty_state() == DirtyState.UNHANDLED_SUBMIT:
        set_dirty_state(DirtyState.DIRTY)
        return True

    return False

enable_custom = True
# Initialize agent
mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

with st.form(key="form"):
    if not enable_custom:
        "Ask one of the sample questions, or enter your API Key in the sidebar to ask your own custom questions."
    prefilled = st.selectbox("Sample questions", sorted(SAVED_SESSIONS.keys())) or ""
    user_input = ""

    if enable_custom:
        user_input = st.text_input("Or, ask your own question")
    if not user_input:
        user_input = prefilled
    submit_clicked = st.form_submit_button("Submit Question")

output_container = st.empty()
if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🦜")
    st_callback = StreamlitCallbackHandler(answer_container)

    # If we've saved this question, play it back instead of actually running LangChain
    # (so that we don't exhaust our API calls unnecessarily)
    if user_input in SAVED_SESSIONS:
        session_name = SAVED_SESSIONS[user_input]
        session_path = Path(__file__).parent / "runs" / session_name
        print(f"Playing saved session: {session_path}")
        answer = playback_callbacks([st_callback], str(session_path), max_pause_time=2)
    else:
        answer = mrkl.run(user_input, callbacks=[st_callback])

    answer_container.write(answer)
