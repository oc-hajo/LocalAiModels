import gpt4all
import asyncio


from langchain_community.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama


def generate_stream(prompt: str, max_tokens: int):
    """
    Async generator to stream response from GPT4All.
    """
    # There are many CallbackHandlers supported, such as
    # from langchain.callbacks.streamlit import StreamlitCallbackHandler

    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate.from_template(template)
    
    callbacks = [StreamingStdOutCallbackHandler()]
    model = GPT4All(model="/root/.cache/gpt4all/Llama-3.2-3B-Instruct-Q4_0.gguf", streaming=True, callbacks=callbacks)

    model = ChatOllama(model='llama3.2:3b-instruct-q4_0', streaming=True)

    chain = prompt | model

    question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

    # Generate text. Tokens are streamed through the callback manager.
    for chunk in chain.stream({"question": question}):
        print(chunk)



if __name__ == "__main__":
    print("generating")
    data = generate_stream("wer bist du?", 123)
    print("done")