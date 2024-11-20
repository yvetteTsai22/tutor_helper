import tiktoken
import json
import re
from tutor_helper.common.llms import LlmLoader


class LlmUtilities:
    def __init__(self):
        pass

    def count_tokens(*_args):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens_total = 0

        for arg in _args:
            token_count = len(encoding.encode(str(arg)))
            tokens_total += token_count

        return int(tokens_total)

    def trim_string_to_token_count(string, max_token_count, trim_iter_size=1750):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        string_original_length = len(encoding.encode(str(string)))

        if string_original_length > max_token_count:
            print(
                f"Trimming original string with {string_original_length} token to max {max_token_count}."
            )
            while len(encoding.encode(str(string))) > max_token_count:
                # TOASK: Why trim_iter_size is 1750?
                # This is to trim the last trim_iter_size characters from the string
                string = string[:-trim_iter_size]  # Trim the last N characters

            # Replace last "cut" word with "..."
            string = re.sub(r"\s+\S+$", "...", string)

            new_length = len(encoding.encode(str(string)))
            print(f"New string token count: {new_length}")

        return string

    def trim_string_to_token_count_new(string: str, max_token_count: int) -> str:
        """To trim token count to max_token_count

        Args:
            string (str): the original string
            max_token_count (int): maximum token count

        Returns:
            str: _description_
        """
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        string_original_length = len(encoding.encode(str(string)))

        if string_original_length > max_token_count:
            print(
                f"Trimming original string with {string_original_length} token to max {max_token_count}."
            )
            string = encoding.decode(encoding.encode(string)[:max_token_count])
            string = re.sub(r"\s+\S+$", "...", string)

            new_length = len(encoding.encode(str(string)))
            print(f"New string token count: {new_length}")

        return string

    def verify_content_is_json(response):
        response_content = response.content
        regexp_beginning = re.compile(r'^```json\s*\{\s*"content":\s*"')
        regexp_end = re.compile(r'"?\s*\}\s*```')

        if not re.search(regexp_beginning, response_content) and not re.search(
            regexp_end, response_content
        ):
            print(
                "[verify_content_is_json]: Beginning and End of response look odd... Wrapping everything in JSON!"
            )
            print(response_content)
            new_response = {"content": response_content}
            response.content = "```json" + json.dumps(new_response) + "```"

        if re.search(regexp_beginning, response_content) and not re.search(
            regexp_end, response_content
        ):
            print(
                "[verify_content_is_json]: Beginning OK, but End of response looks odd... Fixing it!"
            )
            print(response_content)
            # Trim double quote from the end via regexp
            response_content = re.sub('[\\"]*$', "", response_content)
            response.content = response.content + '"}```'

    def get_llm_model_by_token(token_count: int):
        if token_count < 3500:
            return LlmLoader.DEPLOYMENT_35_TURBO
        if token_count < 14000:
            return LlmLoader.DEPLOYMENT_35_TURBO_LG
        return LlmLoader.DEPLOYMENT_GPT4_LG
