import json
import jsonschema
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import re
import tiktoken

default_format = "%(asctime)s\t%(levelname)s\tP%(process)d\tT%(thread)d\t%(filename)s:%(lineno)d\t%(funcName)s: %(message)s"

import logging 
logger = logging.getLogger(__name__)

PathStr = Union[Path, str]
JsonDict = Dict[str, Any]


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def split_by_token(
    string: str, encoding_name: str = "cl100k_base", chunk_length=3000
) -> List[str]:
    num_tokens = num_tokens_from_string(string=string, encoding_name=encoding_name)

    if num_tokens <= chunk_length:
        return [string]
    encoding = tiktoken.get_encoding(encoding_name)
    string_encoding = encoding.encode(string)

    return [
        encoding.decode(string_encoding[i : i + chunk_length])
        for i in range(0, num_tokens, chunk_length)
    ]


def read_json_schema(path: PathStr) -> JsonDict:

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json(path: PathStr, schema: Optional[Union[str, JsonDict]] = None) -> JsonDict:

    """Read the json file at `path` and optionally validates the input according to `schema`.
    The validation requires `jsonschema`.
    `schema` can either be a path as well, or a Python dict which represents the schema.
    `cls` and `object_hook` is passed through to `json.load`.
    """

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if schema is None:
        return obj

    from jsonschema import validate

    if isinstance(schema, str):
        schema = read_json_schema(schema)

    validate(obj, schema)
    return obj


def normalize(name: str) -> str:
    """Simplifies a string for improved string matching."""
    re_norm = re.compile(r"[ ()\-&,]")
    return re_norm.sub("", name).lower()