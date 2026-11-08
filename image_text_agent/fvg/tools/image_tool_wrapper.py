import inspect
import urllib.request

from fvg.tools.ipython import make_output_content, simplify_content
from langchain_core.tools.base import create_schema_from_function
from langchain_core.runnables import RunnableConfig
from langchain.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolRuntime
from PIL import Image
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from typing import Callable
from typing_extensions import override


def url_to_image(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers),
    ) as s:
        image = Image.open(s)
        image.load()

    return image


class ImageToolWrapper(BaseTool):

    name: str = "image_tool_wrapper"
    description: str = (
        "Process the image from the message list with given function."
    )

    underlying_func: Callable
    """The underlying functions to process the image."""

    input_parameters: list
    """The image parameters of the underlying function to convert."""

    image_save_kwargs: dict = {"format": "PNG"}
    """The options to save the image returned in the 'ipython' mode."""

    print_image_object: bool = False
    """Output images as text objects to the text content."""

    @override
    def get_input_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        schema = create_schema_from_function(
            self.name, self.underlying_func, parse_docstring=True)

        # the injected argument to access the message list from the tool
        schema.model_fields["runtime"] = FieldInfo(annotation=ToolRuntime)

        # replace the image parameters with the image index
        for id, i in enumerate(self.input_parameters):
            old_field = schema.model_fields.pop(i)
            new_key = (
                "image_index"
                if len(self.input_parameters) == 1 else
                "image_index_{}".format(id)
            )
            schema.model_fields[new_key] = FieldInfo(
                annotation=int,
                description="The image index of the message contents for the "
                "purpose of '{}'.".format(old_field.description),
            )

        return schema

    @override
    def _run(self, runtime: ToolRuntime, **kwargs):
        sig = inspect.signature(self.underlying_func)
        messages = runtime.state["messages"]
        image_content_list = [
            j["image_url"]["url"]
            for i in messages
            if isinstance(i.content, list)
            for j in i.content
            if j["type"] == "image_url"
        ]
        key_mapping = {
            i: (
                "image_index"
                if len(self.input_parameters) == 1 else
                "image_index_{}".format(id)
            )
            for id, i in enumerate(self.input_parameters)
        }
        func_kwargs = {
            i: (
                url_to_image(image_content_list[kwargs[key_mapping[i]]])
                if i in key_mapping
                else kwargs[i]
            )
            for i in sig.parameters.keys()
        }
        image_result = self.underlying_func(**func_kwargs)
        content = make_output_content(
            None, image_result, self.image_save_kwargs,
            self.print_image_object,
        )
        return simplify_content(content)

    @override
    async def _arun(self, runtime: ToolRuntime, **kwargs):
        return self._run(
            runtime,
            **{
                k: (
                    v.get_sync()
                    if k == "run_manager" and v is not None
                    else v
                )
                for k, v in kwargs.items()
            }
        )
