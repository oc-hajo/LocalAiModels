from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import local_llm
from semantic_search import asymmetric_search, symmetric_search
from starlette.responses import FileResponse 
from pydantic import BaseModel


# Define the input schema
class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 200  # Optional with default

# Initialize FastAPI app
app = FastAPI()






@app.post("/query_stream")
async def query_gpt4all_stream(request: QueryRequest):
    """Endpoint to handle GPT4All queries with streaming."""
    return StreamingResponse(
        local_llm.generate_stream(request.prompt, request.max_tokens),
        media_type="text/plain"
    )

@app.get("/search_sym/{query}")
async def symmetric_semantic_search(query:str):
    return symmetric_search(query)

@app.get("/search_asym/{query}")
async def asymmetric_semantic_search(query:str):
    return asymmetric_search(query)

# Health check endpoint
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')
