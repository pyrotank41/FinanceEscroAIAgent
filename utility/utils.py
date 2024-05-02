
import os
from dotenv import load_dotenv, find_dotenv
from loguru import logger


def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")


def load_prompt(prompt_file: str) -> str:
    if os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            prompt = f.read()
            logger.info(f"Prompt loaded from {prompt_file}")
            return prompt
