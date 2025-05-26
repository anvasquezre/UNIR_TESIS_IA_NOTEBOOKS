from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import torch  # noqa: F401
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from PIL import Image
from unsloth import FastVisionModel

from core.logger import get_logger
from src.extraction.extraction_template import Base64Image

logger = get_logger(__name__)

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)

EOT_TOKEN = "<|eot_id|>"

FastVisionModel.for_inference(model)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_content, images = _format_message_content(message.content)
    message_dict: dict[str, Any] = {"content": message_content}
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "user"
    else:
        raise TypeError(f"Unsupported message type: {type(message)}.")
    return message_dict, images


def _format_message_content(
    content: Any,
) -> Tuple[List[Dict[str, str]], List[Base64Image]]:
    """Format message content."""
    images = []
    if content and isinstance(content, list):
        formatted_content = []
        for block in content:
            # Remove unexpected block types
            if (
                isinstance(block, dict)
                and "type" in block
                and block["type"] in ("tool_use", "thinking")
            ):
                continue
            elif isinstance(block, dict) and block.get("type") == "image":
                # Return only
                formatted_content.append(
                    {
                        "type": "image",
                    }
                )
                if image_data := block.get("data"):
                    images.append(image_data)

                continue
            else:
                formatted_content.append(block)
    else:
        formatted_content = [
            {
                "type": "text",
                "text": content,
            }
        ]

    return formatted_content, images


class LLama3_2_11B_V(BaseChatModel):
    """Llama 3.2 11B V model."""

    model_name: str = "llama-3.2-11B-V"
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 512
    gen_model: Any = model
    tokenizer: Any = tokenizer

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:

        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        messages = payload["messages"]
        images: List[Base64Image] = payload["images"]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        if images:
            images_pil = [
                Image.open(BytesIO(base64.b64decode(image))) for image in images
            ]

            inputs = tokenizer(
                images_pil,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")
        else:
            inputs = tokenizer(
                None,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")

        message_generation = self.gen_model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            use_cache=True,
            temperature=self.temperature,
            min_p=1 - self.top_p,
        )
        message_decoded: str = self.tokenizer.decode(
            message_generation[0],
        )

        # remove input text from the output
        message_decoded = message_decoded[len(input_text) :]
        # remove EOT token
        message_decoded = message_decoded.replace(EOT_TOKEN, "")
        # remove any leading or trailing whitespace
        message_decoded = message_decoded.strip()
        # remove any leading or trailing newlines
        message_decoded = message_decoded.strip("\n")
        logger.info(
            f"Decoded message: {message_decoded}... (length: {len(message_decoded)})"
        )

        message = AIMessage(content=str(message_decoded))
        generations = ChatGeneration(message=message)
        return ChatResult(
            generations=[generations],
            llm_output={
                "model_name": self.model_name,
                "stop": stop,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            },
        )

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "llama-3.2-11B-V"

    def _format_chat_conversation(
        self,
        context: str,
        image: Base64Image,
    ) -> dict[str, List[Dict[str, Any]]]:
        """Format the conversation for the model."""
        return

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:
        payload = {}
        messages = self._convert_input(input_).to_messages()

        payload["messages"] = []
        payload["images"] = []
        for m in messages:
            message, images = _convert_message_to_dict(m)
            if images:
                payload["images"].extend(images)
            payload["messages"].append(message)

        return payload

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
