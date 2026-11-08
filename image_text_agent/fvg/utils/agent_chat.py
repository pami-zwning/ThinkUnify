import argparse
import base64
import filetype
import urllib.request

from fvg.messages.utils import convert_to_openai_messages
from langchain_core.messages import AIMessage
from langgraph.pregel import Pregel
from langgraph.types import Command


class ChuckPrinter:
    def __init__(self):
        self.current_step = 0

    def update(self, message_chunk, metadata):
        if self.current_step != metadata["langgraph_step"]:
            if self.current_step > 0:
                print()

            self.current_step = metadata["langgraph_step"]
            print("{}:".format(metadata["langgraph_node"]), end="")
            if (
                metadata["langgraph_node"] == "model" and
                "tags" in metadata and metadata["tags"] is not None
            ):
                for i in metadata["tags"]:
                    print("{}:".format(i), end="")

        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)

        if hasattr(message_chunk, "tool_calls"):
            for i in message_chunk.tool_calls:
                if i["name"]:
                    print("\n->{}:".format(i["name"]), end="")

                if i["args"]:
                    print(i["args"], end="")


def print_message(node: str, messages: list):
    for message in messages:
        print("{}:{}".format(node, message.content))
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_call_content = "->{}: {}".format(
                    tool_call["name"], tool_call["args"])
                print(tool_call_content)


def make_user_message_content_list(query_list: str):
    content_list = []
    complex_content = None
    headers = {"User-Agent": "Mozilla/5.0"}
    supported_proxies = ["file:", "data:", "http:", "https:"]
    for query in query_list:
        if any([query.startswith(j) for j in supported_proxies]):
            # supported URI
            with urllib.request.urlopen(
                urllib.request.Request(query, headers=headers),
            ) as s:
                data = s.read()

            if complex_content is None:
                complex_content = []

            mime_type = filetype.guess_mime(data)
            base64_data = base64.b64encode(data).decode("utf-8")
            if mime_type.startswith("image"):
                data_uri = "data:{};base64,{}".format(mime_type, base64_data)
                complex_content.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                })
            else:
                complex_content.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_data,
                    }
                })

        else:
            # text message (1st message is prepended with reference content)
            if complex_content is None:
                content_list.append(query)
            else:
                complex_content.append({"type": "text", "text": query})
                content_list.append(complex_content)
                complex_content = None

    # flush the complex content
    if complex_content is not None:
        content_list.append(complex_content)

    return content_list


async def async_chat_loop(
    config: dict, agent: Pregel, query: list, stream_config: dict,
    stream_mode: str = "messages", input_state=None,
):
    content_list = make_user_message_content_list(query)

    while True:
        if input_state:
            graph_output = agent.astream(
                input_state, stream_config, stream_mode=stream_mode)
            input_state = None

        else:
            content = (
                content_list.pop(0)
                if len(content_list) > 0
                else input("human: ")
            )
            if content == "q":
                break

            interrupts = agent.get_state(stream_config).interrupts
            graph_output = agent.astream(
                (
                    {
                        config.get("messages_key", "messages"): [
                            {"role": "user", "content": content},
                        ],
                    }
                    if len(interrupts) == 0
                    else Command(resume=content)
                ),
                stream_config, stream_mode=stream_mode)

        if stream_mode == "messages":
            printer = ChuckPrinter()
            async for message_chunk, metadata in graph_output:
                printer.update(message_chunk, metadata)

            print()

        elif stream_mode == "updates":
            async for step_output in graph_output:
                for node, response in step_output.items():
                    if node == "__interrupt__":
                        break

                    step_0 = agent.nodes[node].node.steps[0]
                    messages_key = getattr(step_0, "messages_key", "messages")
                    print_message(node, response[messages_key])


def chat_loop(
    config: dict, agent: Pregel, query: list, stream_config: dict,
    input_state=None,
):
    content_list = make_user_message_content_list(query)

    while True:
        if input_state:
            graph_output = agent.stream(
                input_state, stream_config, stream_mode="updates")
            input_state = None

        else:
            content = (
                content_list.pop(0)
                if len(content_list) > 0
                else input("human: ")
            )
            if content == "q":
                break

            interrupts = agent.get_state(stream_config).interrupts
            graph_output = agent.stream(
                (
                    {
                        config.get("messages_key", "messages"): [
                            {"role": "user", "content": content},
                        ],
                    }
                    if len(interrupts) == 0
                    else Command(resume=content)
                ),
                stream_config, stream_mode="updates")

            for step_output in graph_output:
                for node, response in step_output.items():
                    if node == "__interrupt__":
                        break

                    step_0 = agent.nodes[node].node.steps[0]
                    messages_key = getattr(step_0, "messages_key", "messages")
                    print_message(node, response[messages_key])


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to chat with the agent.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The path to the config file.")
    parser.add_argument(
        "-i", "--input-state-path", default=None, type=str,
        help="The path to resume the conversation by the state (if given).")
    parser.add_argument(
        "-o", "--output-path", default=None, type=str,
        help="The path to the save the chat history (if given).")
    parser.add_argument(
        "-q", "--query", nargs="*", help="The begining queries.")
    parser.add_argument(
        "--disable-async", action="store_true", default=False,
        help="Disable the async mode for better compatibility.")
    parser.add_argument(
        "--stream-mode", default="messages", type=str,
        help="The stream mode.")
    return parser


if __name__ == "__main__":
    import asyncio
    import fvg.common
    import json

    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.input_state_path:
        with open(args.input_state_path, "r", encoding="utf-8") as f:
            input_state = json.load(f)
    else:
        input_state = None

    # load models and other framework modules
    agent = fvg.common.make_object_from_config(config["agent"])

    # run the graph
    stream_config = config["stream_config"].copy()
    try:
        if args.disable_async:
            chat_loop(
                config, agent, args.query or [], stream_config,
                input_state=input_state,
            )
        else:
            asyncio.run(
                async_chat_loop(
                    config, agent, args.query or [], stream_config,
                    stream_mode=args.stream_mode, input_state=input_state,
                ))
    except Exception as e:
        print(e)

    # save the result
    if args.output_path:
        state_values = agent.get_state(stream_config).values
        output_data = {
            i: (
                state_values[i].copy()
                if isinstance(state_values[i], dict)
                else (
                    convert_to_openai_messages(
                        state_values[i], ensure_ascii=False)
                    if isinstance(state_values[i], list)
                    else state_values[i]
                )
            )
            for i in config.get("collect_output_from", ["messages"])
            if i in state_values
        }
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
