import asyncio
import os

from openai import AsyncOpenAI


async def check_health(
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    disable_http_proxy: bool = True,
):
    """
    Check if the OpenAI client can connect to the API server.

    Returns:
        bool: True if the client is healthy, False otherwise.
    """
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    if disable_http_proxy:
        os.environ.pop("http_proxy", None)
    try:
        # List models is a lightweight API call that can be used as a health check
        models = await client.models.list()
        print(
            f"✅ Connection successful! Available models: {[model.id for model in models.data]}"
        )
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(check_health())
