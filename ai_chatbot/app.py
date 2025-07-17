from flask import Flask, render_template, request
from typing_extensions import TypedDict
from typing import Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
import atexit

load_dotenv()
app = Flask(__name__)

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = init_chat_model(model_provider="openai", model="gpt-4.1")

def chat_node(state: State):
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

# Global variables
graph = None
mongo_checkpointer_ctx = None  # the context manager
mongo_checkpointer = None      # the actual checkpointer

def compile_graph_with_checkpointer(checkpointer):
    graph_builder = StateGraph(State)
    graph_builder.add_node("chat_node", chat_node)
    graph_builder.add_edge(START, "chat_node")
    graph_builder.add_edge("chat_node", END)
    return graph_builder.compile(checkpointer=checkpointer)

def init_graph():
    global graph, mongo_checkpointer_ctx, mongo_checkpointer
    DB_URI = "mongodb://admin:admin@mongodb:27017"

    # Store context manager and enter it
    mongo_checkpointer_ctx = MongoDBSaver.from_conn_string(DB_URI)
    mongo_checkpointer = mongo_checkpointer_ctx.__enter__()

    # Register cleanup
    atexit.register(lambda: mongo_checkpointer_ctx.__exit__(None, None, None))

    graph = compile_graph_with_checkpointer(mongo_checkpointer)

init_graph()

# In-memory chat history
chat_history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history
    if request.method == "POST":
        user_input = request.form["user_input"]
        config = {"configurable": {"thread_id": "1"}}

        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]}, config
        )

        response = result["messages"][-1].content
        chat_history.append(("You : ", user_input))
        chat_history.append(("Bot 🤖 : ", response))

    return render_template("index.html", history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)
