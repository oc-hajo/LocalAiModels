from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse 
from pydantic import BaseModel
import local_llm
import local_llm_embeddings
import semantic_search

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
    return semantic_search.symmetric_search(query)

@app.get("/search_asym/{query}")
async def asymmetric_semantic_search(query:str):
    return semantic_search.asymmetric_search(query)

@app.post("/document_search")
async def query_documents(request: QueryRequest):
    return StreamingResponse(
        local_llm_embeddings.document_search(request.prompt),
        media_type="text/plain"
    )

# Health check endpoint
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')
