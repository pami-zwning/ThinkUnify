import json
import re
import transformers
import uuid

from collections.abc import Callable, Sequence
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, ToolCall, convert_to_openai_messages
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from typing import Any
from typing_extensions import override


def openai_messages_to_conversation(
    model_class_name: str, openai_messages: list,
) -> list:
    if model_class_name.startswith("Qwen3VL"):
        return [
            {
                "role": i["role"],
                "content": [
                    {
                        "type": "image",
                        "image": j["image_url"]["url"],
                    } if j["type"] == "image_url" else j
                    for j in i["content"]
                ]
            } if isinstance(i["content"], list) else i
            for i in openai_messages
        ]
    else:
        return openai_messages


def make_chat_result(
    model_class_name: str, generated_text: str, eos_token: str,
) -> ChatResult:
    fallback_message = AIMessage(generated_text)
    if model_class_name.startswith("Qwen3VL"):
        try:
            content_match = list(re.finditer(
                r"^((?P<reasoning_content>.*)</think>)?(?P<content>.*?)(" +
                re.escape(eos_token) + ")*$",
                generated_text, re.DOTALL,
            ))[0]
            reasoning_content = content_match.group("reasoning_content")
            content = content_match.group("content")
            tool_call_matches = re.finditer(
                r"<tool_call>(?P<tool_call_content>.*?)</tool_call>",
                content, re.DOTALL,
            )
            tool_calls = []
            remained_content = ""
            previous_end = 0
            for i in tool_call_matches:
                remained_content += content[previous_end:i.start()]
                previous_end = i.end()

                tool_call_content = i.group("tool_call_content")
                for j in tool_call_content.strip("\n").split("\n"):
                    line = j.strip("\n")
                    try:
                        tool_call = json.loads(line)
                        tool_calls.append(tool_call)
                    except:
                        pass  # ignore invalid tool call

            remained_content += content[previous_end:]
            message = AIMessage(
                remained_content.strip("\n"),
                additional_kwargs={
                    "reasoning_content": reasoning_content,
                },
                tool_calls=[
                    ToolCall(
                        name=i["name"], args=i["arguments"],
                        id=uuid.uuid4().hex, type="tool_call",
                    )
                    for i in tool_calls
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=message)])

        except:
            pass

    return ChatResult(
        generations=[ChatGeneration(message=fallback_message)])


class ChatTransformers(BaseChatModel):

    model_class_name: str
    model_args: dict
    model: Any = None
    processor_class_name: str
    processor_args: dict
    processor: Any = None
    max_new_tokens: int = 4096

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        model_class = getattr(transformers, self.model_class_name)
        self.model = model_class.from_pretrained(**self.model_args)
        processor_class = getattr(transformers, self.processor_class_name)
        self.processor = processor_class.from_pretrained(**self.processor_args)

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        conversation = openai_messages_to_conversation(
            self.model_class_name, convert_to_openai_messages(messages),
        )
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True,
            add_generation_prompt=True, return_tensors="pt", **kwargs,
        ).to(self.model.device)
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        return make_chat_result(
            self.model_class_name,
            self.processor.batch_decode(generated_ids_trimmed)[0],
            self.processor.tokenizer.special_tokens_map["eos_token"],
        )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        assert tool_choice is None, "tool_choice is not supported"

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    @override
    def _llm_type(self) -> str:
        return "transformers-chat-wrapper"
