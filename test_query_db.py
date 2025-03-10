from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain.chains import create_sql_query_chain

from dotenv import load_dotenv
load_dotenv()

mysql_uri = 'mysql+mysqlconnector://root:admin@localhost:3306/oie'
db = SQLDatabase.from_uri(mysql_uri)
print("dialect:", db.dialect)
# print("table name:", db.get_usable_table_names())
# print("table_info:", db.get_table_info())
# print(db.run("SELECT * FROM agencies LIMIT 10;"))

from typing_extensions import TypedDict

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langchain import hub

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

assert len(query_prompt_template.messages) == 1
# query_prompt_template.messages[0].pretty_print()

from typing_extensions import Annotated

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

# print(write_query({"question": "แสดง agency ที่ ministry_id=1 แค่ 10 แถวแรก"}))

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDataBaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

# print(execute_query({"query": "SELECT id, code, name_th, name_en FROM agencies WHERE ministry_id = 1 LIMIT 10;"}))

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

for step in graph.stream(
    {"question": "แสดง agency ที่ ministry_id=1 แค่ 10 แถวแรก"}, 
    stream_mode="updates",
):
    print(step)
