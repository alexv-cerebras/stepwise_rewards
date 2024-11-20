from openai.types.chat.chat_completion import ChatCompletion

import os
import openai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
OPENAI_BASE_URL = None


def get_openai_api_key() -> str:
    return os.environ["OPENAI_API_KEY"]

def _get_openai_client(base_url = None) -> openai.Client:
    api_key = get_openai_api_key()    
    print('api_key: ', api_key)
    return openai.Client(api_key=api_key, base_url=base_url)


@retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=1, min=4, max=10))
def openai_chat_completion(
    messages: list,
    model: str,
    client: any,
    temperature: float = 0.8,
    **kwargs,
) -> ChatCompletion:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response
    