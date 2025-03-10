import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain import hub
from typing_extensions import TypedDict
from typing_extensions import Annotated
from urllib.parse import quote

import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY_AN"]

from dotenv import load_dotenv
load_dotenv()

# app config
st.set_page_config(page_title="ระบบสอบถามข้อมูลจาก Database", page_icon="🤖")
chat_ai_icon = "icon/authoritative_government_officer.png"
chat_user_icon = "icon/user.png"

# connect to database
def connect_database(db_name: str):
    host = "192.168.10.50"
    port = "3306"
    username = st.secrets["DB_USERNAME"]
    password = st.secrets["DB_PASSWORD"]
    mysql_uri = f"mysql+mysqlconnector://{username}:%s@{host}:{port}/{db_name}" % quote(f"{password}")
    db = SQLDatabase.from_uri(mysql_uri)

    return db

db = connect_database(db_name="oie")
writing_sql_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_sql_query(question: str, db: SQLDatabase) -> str:
    """Generate SQL query to fetch information."""
    query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
    sql_prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 50,
            "table_info": db.get_table_info(),
            "input": question,
        }
    )
    structured_llm = writing_sql_llm.with_structured_output(QueryOutput)
    sql_query = structured_llm.invoke(sql_prompt)

    return sql_query

def execute_query(sql_query: str, db: SQLDatabase) -> str:
    """Execute SQL query."""
    execute_query_tool = QuerySQLDataBaseTool(db=db)
    sql_result = execute_query_tool.invoke(sql_query)

    return sql_result

def create_chain():
    """Answer question using retrieved information as context."""
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are a man working as an authoritative Thai government officer.
            Given the following SQL query and SQL result, answer the user question.

            Condition:
            - Do not use any other information to answer the question.
            - Only answer question about data in the database.
            - The answers mustn't contain anything about SQL.
            - If the SQL Result is null, just reply I don't know.
            - If the SQL Result doesn't correspond to the question, just frankly reply I can't answer the question because of no corresponding context.

            'SQL Query: ```{sql_query}```'
            'SQL Result: ```{sql_result}```'"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chain = response_prompt | response_llm | StrOutputParser()

    return chain

def get_response(question, chat_history):
    # 1. get an appropriate SQL query based on the user's query
    sql_query = write_sql_query(question=question, db=db)
    print("SQL query:", sql_query) # debug

    # 2. get query result from the database
    sql_result = execute_query(sql_query=sql_query, db=db)

    # 3. answer by natural language
    chain = create_chain()
    return chain.stream({
          "question" : question,
          "chat_history": chat_history,
          "sql_query": sql_query,
          "sql_result": sql_result
    })

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="สวัสดีครับ ผมคือเจ้าหน้าที่ AI ทำหน้าที่ตอบคำถามจากฐานข้อมูลในองค์กรครับ"),
    ]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar=chat_ai_icon):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar=chat_user_icon):
            st.write(message.content)

# user input
user_query = st.chat_input("พิมพ์คำถามตรงนี้ ...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human", avatar=chat_user_icon):
        st.markdown(user_query)

    with st.chat_message("AI", avatar=chat_ai_icon):
        response = get_response(user_query, st.session_state.chat_history[-6:])
        response = st.write_stream(response)

        st.session_state.chat_history.append(AIMessage(content=response))
