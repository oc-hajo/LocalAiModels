import gpt4all
import asyncio

async def generate_stream(prompt: str, max_tokens: int):
    """
    Async generator to stream response from GPT4All.
    """
    # Load the GPT4All model
    try:
        model = gpt4all.GPT4All('Phi-3-mini-4k-instruct.Q4_0.gguf', n_threads=8, allow_download=True)
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None

    if model is None:
        yield "[Error: Model not loaded]"
        return

    try:
        with model.chat_session():
            # Generate response in chunks
            for chunk in model.generate(prompt, max_tokens=max_tokens, streaming=True):
                yield chunk
                await asyncio.sleep(0)  # Allow other coroutines to run
    except Exception as e:
        yield f"[Error: {e}]"