# Example function that embeds a single chunk using Azure OpenAI
from openai import AzureOpenAI
from typing import List
import os
from dotenv import load_dotenv
from gen_utils.parsing_utils import load_config
from utils import store_secret


def embed_texts_in_batch(text_batch: List[str]) -> List[List[float]]:
    # your existing approach
    config = load_config("./configs/config.json")

    store_secret(
    secret_name=config['secret_name'],
    project_id=config['project_id'],
    )

    # # Client for Embeddings
    client_embedding = AzureOpenAI(
            azure_endpoint="https://data-ai-labs.openai.azure.com/",
            api_key=os.getenv("AZURE1"),
            api_version="2024-08-01-preview",
        )

    response = client_embedding.embeddings.create(
        model="embed-3-large-auto",
        input=text_batch
    )
    return [item.embedding for item in response.data]

# Example function that embeds a single chunk using Azure OpenAI
def embed_chunk(text: str) -> List[float]:

    # your existing approach
    config = load_config("./configs/config.json")

    store_secret(
    secret_name=config['secret_name'],
    project_id=config['project_id'],
    )

    # # Client for Embeddings
    client_embedding = AzureOpenAI(
            azure_endpoint="https://data-ai-labs.openai.azure.com/",
            api_key=os.getenv("AZURE1"),
            api_version="2024-08-01-preview",
        )

    response = client_embedding.embeddings.create(
        model="embed-3-large-auto",
        input=[text]
    )
    # Azure’s response typically has a 'data' array with 'embedding' inside
    embedding_vector = response.data[0].embedding
    return embedding_vector
