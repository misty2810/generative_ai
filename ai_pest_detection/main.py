from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import base64
import openai
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph
import asyncio

# Load OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in your .env file.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

class PestState(TypedDict):
    image_b64: str
    description: str
    diagnosis: str

def describe_leaf(state: PestState) -> Dict[str, Any]:
    """
    Use GPT-4.1/GPT-4o to visually describe the uploaded leaf image.
    Returns a description or error.
    """
    if "image_b64" not in state or not state["image_b64"]:
        return {"error": "No image found in input."}
    prompt = (
        "You are a plant pathology expert. Describe all visible features of this leaf: "
        "color changes, spots, holes, edge damage, fungus, or insect trails. Do not guess a diagnosis yet."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{state['image_b64']}" }}
            ]
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4.1", 
            messages=messages,
            max_tokens=800
        )
        desc = response.choices[0].message.content or ""
        return {"description": desc}
    except Exception as e:
        print("ERROR: GPT-4.1/4o", e)
        return {"error": f"GPT-4.1/4o error: {str(e)}"}

def diagnose_leaf(state: PestState) -> Dict[str, Any]:
    """
    Use GPT-4.1/Turbo/4o to analyze the leaf description and suggest disease/pest and treatment.
    Returns a diagnosis or error.
    """
    description = state.get("description", "")
    if not description:
        return {"error": "No description provided for diagnosis."}
    prompt = (
        f"You are a plant pathology expert. Given the following description of a plant leaf, answer:\n"
        "1. What symptoms are visible?\n"
        "2. What is the most likely disease?\n"
        "3. What pest or pathogen might cause it?\n"
        "4. What are recommended organic and chemical treatments?\n"
        f"\nDescription:\n{description}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4.1", 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        diag = response.choices[0].message.content or ""
        return {"diagnosis": diag}
    except Exception as e:
        print("ERROR: GPT-4.1/4o", e)
        return {"error": f"GPT-4.1/4o error: {str(e)}"}

# LangGraph pipeline setup
graph = StateGraph(PestState)
graph.add_node("describe_leaf", describe_leaf)
graph.add_node("diagnose_leaf", diagnose_leaf)
graph.set_entry_point("describe_leaf")
graph.add_edge("describe_leaf", "diagnose_leaf")
graph.set_finish_point("diagnose_leaf")
flow = graph.compile()

@app.get("/")
def root():
    return {"message": "Upload a leaf image via POST /analyze (JPEG or PNG only)."}

@app.post("/analyze")
async def analyze_leaf(file: UploadFile = File(...)):
    print(f"DEBUG: Input content_type: {file.content_type}")
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    b64_image = base64.b64encode(contents).decode()
    initial_state: PestState = {
        "image_b64": b64_image,
        "description": "",
        "diagnosis": ""
    }
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, flow.invoke, initial_state)
    desc = result.get("description") or result.get("error", "")
    diag = result.get("diagnosis", "")
    if not desc and "error" in result:
        desc = result["error"]
    return JSONResponse({
        "message": "Analysis complete.",
        "filename": file.filename,
        "description": desc,
        "diagnosis": diag
    })