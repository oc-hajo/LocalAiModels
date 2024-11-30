FROM python:3.12

WORKDIR /app

RUN pip install gpt4all fastapi pydantic
RUN python -c "from gpt4all import GPT4All; GPT4All('Phi-3-mini-4k-instruct.Q4_0.gguf')"

RUN pip install fastapi[standard]
RUN pip install sentence_transformers
RUN pip install langchain chromadb langchain-community
RUN pip install pypdf
RUN python -c "from gpt4all import GPT4All; GPT4All('Llama-3.2-3B-Instruct-Q4_0.gguf')"


RUN apt update && apt install htop -y

COPY ./src/server.py .
CMD ["fastapi", "run", "server.py"]
