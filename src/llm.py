import logging
from langchain_core.documents import Document
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
import logging
from langchain_core.documents import Document
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_experimental.graph_transformers.llm import _Graph
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_community.chat_models import ChatOllama
import google.auth
from src.adaptive_retry import resolve_graph_transformer_settings
from src.shared.constants import ADDITIONAL_INSTRUCTIONS
from src.shared.llm_graph_builder_exception import LLMGraphBuilderException
import re
from typing import List
from langchain_core.callbacks.manager import CallbackManager
from src.shared.common_fn import UniversalTokenUsageHandler, get_value_from_env


def get_llm(model: str):

        elif "GOOGLE" in model:
            # Google Gemini via OpenAI-compatible endpoint using GOOGLE_API_KEY env var
            # LLM_MODEL_CONFIG_google_<name>="model_name" (api key read from GOOGLE_API_KEY)
            model_name = env_value.strip().split(
                ",")[0]  # only model name needed
            google_api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not google_api_key:
                raise Exception(
                    "GOOGLE_API_KEY environment variable is not set")
            llm = ChatOpenAI(
                api_key=google_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                model=model_name,
                temperature=0,
                callbacks=callback_manager,
            )

        elif "DIFFBOT" in model:
            # model_name = "diffbot"
            model_name, api_key = env_value.split(",")
            llm = DiffbotGraphTransformer(
                diffbot_api_key=api_key,
                extract_types=["entities", "facts"],
            )
            callback_handler = None

        else:
            model_name, api_endpoint, api_key = env_value.split(",")
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=api_endpoint,
                model=model_name,
                temperature=0,
                callbacks=callback_manager,
            )
    except Exception as e:
        err = f"Error while creating LLM '{model}': {str(e)}"
        logging.error(err)
        raise Exception(err)

    logging.info(f"Model created - Model Version: {model}")
    return llm, model_name, callback_handler


def get_llm_model_name(llm):
    """Extract name of llm model from llm object"""
    for attr in ["model_name", "model", "model_id"]:
        model_name = getattr(llm, attr, None)
        if model_name:
            return model_name.lower()
    logging.info("Could not determine model name; defaulting to empty string")
    return ""


def get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine):
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list


def get_chunk_id_as_doc_metadata(chunkId_chunkDoc_list):
    combined_chunk_document_list = [
        Document(
            page_content=document["chunk_doc"].page_content,
            metadata={"chunk_id": [document["chunk_id"]]},
        )
        for document in chunkId_chunkDoc_list
    ]
    return combined_chunk_document_list
async def get_graph_document_list(
    llm,
    combined_chunk_document_list,
    allowedNodes,
    allowedRelationship,
    callback_handler,
    additional_instructions=None,
    low_verbosity: bool = False,
):
    if additional_instructions:
        additional_instructions = sanitize_additional_instruction(
            additional_instructions)
    graph_document_list = []
    token_usage = 0
    try:
        if "diffbot_api_key" in dir(llm):
            llm_transformer = llm
        else:
            try:
                llm.with_structured_output(_Graph)
                supports_structured_output = True
            except Exception:
                supports_structured_output = False
            settings = resolve_graph_transformer_settings(
                supports_structured_output=supports_structured_output,
                is_groq=isinstance(llm, ChatGroq),
                low_verbosity=low_verbosity,
            )
            node_properties = settings["node_properties"]
            relationship_properties = settings["relationship_properties"]
            ignore_tool_usage = settings["ignore_tool_usage"]
            if settings["mode"] == "low_verbosity":
                logging.info(
                    "Low-verbosity graph extraction enabled; excluding descriptions and structured tool usage")
            elif settings["mode"] == "structured":
                logging.info(
                    "LLM supports structured output; including descriptions in graph")
            else:
                logging.info(
                    "LLM does not support structured output; excluding descriptions in graph")

            llm_transformer = LLMGraphTransformer(
                llm=llm,
                node_properties=node_properties,
                relationship_properties=relationship_properties,
                allowed_nodes=allowedNodes,
                allowed_relationships=allowedRelationship,
                ignore_tool_usage=ignore_tool_usage,
                additional_instructions=ADDITIONAL_INSTRUCTIONS +
                (additional_instructions if additional_instructions else "")
            )

        if isinstance(llm, DiffbotGraphTransformer):
            graph_document_list = llm_transformer.convert_to_graph_documents(
                combined_chunk_document_list)
        else:
            graph_document_list = await llm_transformer.aconvert_to_graph_documents(combined_chunk_document_list)
    except Exception as e:
        logging.error(f"Error in graph transformation: {e}", exc_info=True)
        raise LLMGraphBuilderException(
            f"Graph transformation failed: {str(e)}")
    finally:
        try:
            if callback_handler:
                usage = callback_handler.report()
                token_usage = usage.get("total_tokens", 0)
        except Exception as usage_err:
            logging.error(f"Error while reporting token usage: {usage_err}")

    return graph_document_list, token_usage


async def get_graph_from_llm(
    model,
    chunkId_chunkDoc_list,
    allowedNodes,
    allowedRelationship,
    chunks_to_combine,
    additional_instructions=None,
    low_verbosity: bool = False,
):
    try:
        llm, model_name, callback_handler = get_llm(model)
        logging.info(f"Using model: {model_name}")

        combined_chunk_document_list = get_combined_chunks(
            chunkId_chunkDoc_list, chunks_to_combine)
        logging.info(f"Combined {len(combined_chunk_document_list)} chunks")

        if allowedNodes:
            allowed_nodes = [node.strip()
                             for node in allowedNodes.split(',') if node.strip()]
        else:
            allowed_nodes = []
        logging.info(f"Allowed nodes: {allowed_nodes}")

        allowed_relationships = []
        if allowedRelationship:
            items = [item.strip()
                     for item in allowedRelationship.split(',') if item.strip()]
            if len(items) % 3 != 0:
                raise LLMGraphBuilderException(
                    "allowedRelationship must be a multiple of 3 (source, relationship, target)")
            for i in range(0, len(items), 3):
                source, relation, target = items[i:i + 3]
                if source not in allowed_nodes or target not in allowed_nodes:
                    raise LLMGraphBuilderException(
                        f"Invalid relationship ({source}, {relation}, {target}): "
                        f"source or target not in allowedNodes"
                    )
                allowed_relationships.append((source, relation, target))
            logging.info(f"Allowed relationships: {allowed_relationships}")
        else:
            logging.info("No allowed relationships provided")

        graph_document_list, token_usage = await get_graph_document_list(
            llm,
            combined_chunk_document_list,
            allowed_nodes,
            allowed_relationships,
            callback_handler,
            additional_instructions,
            low_verbosity=low_verbosity,
        )
        logging.info(f"Generated {len(graph_document_list)} graph documents")
        return graph_document_list, token_usage
    except Exception as e:
        logging.error(f"Error in get_graph_from_llm: {e}", exc_info=True)
        raise LLMGraphBuilderException(f"Error in getting graph from llm: {e}")


def sanitize_additional_instruction(instruction: str) -> str:
    """
    Sanitizes additional instruction by:
    - Replacing curly braces `{}` with `[]` to prevent variable interpretation.
    - Removing potential injection patterns like `os.getenv()`, `eval()`, `exec()`.
    - Stripping problematic special characters.
    - Normalizing whitespace.
    Args:
        instruction (str): Raw additional instruction input.
    Returns:
        str: Sanitized instruction safe for LLM processing.
    """
    logging.info("Sanitizing additional instructions")
    # Convert `{}` to `[]` for safety
    instruction = instruction.replace("{", "[").replace("}", "]")
    # Step 2: Block dangerous function calls
    injection_patterns = [
        r"os\.getenv\(", r"eval\(", r"exec\(", r"subprocess\.", r"import os", r"import subprocess"]
    for pattern in injection_patterns:
        instruction = re.sub(
            pattern, "[BLOCKED]", instruction, flags=re.IGNORECASE)
    # Step 4: Normalize spaces
    instruction = re.sub(r'\s+', ' ', instruction).strip()
    return instruction
