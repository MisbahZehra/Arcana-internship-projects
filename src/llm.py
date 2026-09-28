"""Configure the Groq chat model from local environment settings."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class LLMConfigurationError(RuntimeError):
	"""Raised when the Groq model configuration is incomplete."""


def get_llm() -> ChatGroq:
	"""Create a low-temperature Groq chat model using local environment values."""
	api_key = os.getenv("GROQ_API_KEY")
	if not api_key:
		raise LLMConfigurationError(
			"GROQ_API_KEY is not configured. Add it to the project .env file."
		)

	model_name = os.getenv("GROQ_MODEL")
	if not model_name:
		raise LLMConfigurationError(
			"GROQ_MODEL is not configured. Add a currently supported Groq model "
			"to the project .env file."
		)

	return ChatGroq(
		model=model_name,
		api_key=api_key,
		temperature=0.1,
	)
