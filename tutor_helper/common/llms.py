from langchain_community.chat_models import AzureChatOpenAI
from langchain.callbacks.manager import Callbacks
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI, AzureOpenAI

from tutor_helper.callbacks.stdout_all import StdOutAllCallbackHandler

from typing import Optional

import os

import logging
logger = logging.getLogger(__name__)


class LlmLoader:
    
    DEPLOYMENT_35_TURBO= "gpt-35-turbo"
    DEPLOYMENT_35_TURBO_LG= "gpt-35-turbo-16k"
    DEPLOYMENT_GPT4_STD= "gpt-4"
    DEPLOYMENT_GPT4_LG= "gpt-4-32k"

    TOKEN_LIMITS = {
        "DEPLOYMENT_35_TURBO": 4000,
        "DEPLOYMENT_35_TURBO_LG": 16000,
        "DEPLOYMENT_GPT4_STD": 128000,
        "DEPLOYMENT_GPT4_LG": 32000,
    }

    TOKEN_LIMITS_ = {
        "gpt-35-turbo": 4000,
        "gpt-35-turbo-16k": 16000,
        "gpt-4": 128000,
        "gpt-4-32k": 32000,
    }

    """Creates an instance to communicate with LLM API."""

    @staticmethod
    def create_llm(
        model: Optional[str] = DEPLOYMENT_35_TURBO,
        temperature: Optional[float] = 0,
        top_p: Optional[float] = 0.9,
        request_timeout_seconds: Optional[int] = 40,
        max_retries: Optional[int] = 2,
        api_version="2024-02-01",
        callbacks: Callbacks = [StdOutAllCallbackHandler()],
        verbose=False,
    ):

        # Creating LLM instance based on configurations
        if os.getenv("OPENAI_API_TYPE") == "azure":
            random_openai_api_base, random_openai_api_key = os.getenv(f"OPENAI_API_BASE"), os.getenv(f"OPENAI_API_KEY")
            logger.info( f"[llms.create_llm] - Using Azure API: {random_openai_api_base}" )
            print( f"[llms.create_llm] - Using Azure API: {random_openai_api_base} with model [{model}]" )
            return AzureOpenAI(
                callbacks=callbacks,
                azure_deployment=model,
                azure_endpoint=random_openai_api_base,
                max_retries=max_retries,
                openai_api_key=random_openai_api_key,
                openai_api_version=api_version,
                request_timeout=request_timeout_seconds,
                temperature=temperature,
                verbose=verbose,
                model_kwargs={
                    "top_p": top_p,
                },
            )
        else:
            return OpenAI(
                temperature=temperature, callbacks=callbacks, request_timeout=1200
            )

    """Creates a instance to communicate with LLM API."""

    @staticmethod
    def create_chat_llm(
        model: Optional[str] = DEPLOYMENT_35_TURBO,
        temperature: Optional[float] = 0.5,
        top_p: Optional[float] = 0.5,
        request_timeout_seconds: Optional[int] = 40,
        max_retries: Optional[int] = 2,
        api_version="2024-02-01",
        callbacks: Callbacks = [StdOutAllCallbackHandler()],
        verbose=False,
        **kwargs,
    ):
        # Creating LLM instance based on configurations
        if os.getenv("OPENAI_API_TYPE") == "azure":
            random_openai_api_base, random_openai_api_key = os.getenv(f"OPENAI_API_BASE"), os.getenv(f"OPENAI_API_KEY")            
            logger.info( f"[llms.create_chat_llm] - Using Azure API: {random_openai_api_base}" )
            print( f"[llms.create_llm] - Using Azure API: {random_openai_api_base} with model [{model}]" )
            return AzureChatOpenAI(
                callbacks=callbacks,
                azure_deployment=model,
                azure_endpoint=random_openai_api_base,
                max_retries=max_retries,
                openai_api_key=random_openai_api_key,
                openai_api_version=api_version,
                request_timeout=request_timeout_seconds,
                temperature=temperature,
                verbose=verbose,
                model_kwargs={
                    "top_p": top_p,
                },
                **(kwargs or {}),
            )
        else:
            return OpenAI(
                temperature=temperature, callbacks=callbacks, request_timeout=request_timeout_seconds
            )

    @staticmethod
    def create_chain(
        prompt: PromptTemplate,
        callbacks: Callbacks = None,
        **kwargs
    ) -> LLMChain:
        """User friendly way to initialize the LLM.
        Args:
            prompt:The PromptTemplate object to be used by the LLMChain
            **kwargs: parameters to be passed to llm initialization function.
        Returns:
            An initialized LLMChain object.
        """
        llm = LlmLoader.create_llm(callbacks=callbacks, **kwargs)
        return LLMChain(llm=llm, prompt=prompt, callbacks=callbacks)
