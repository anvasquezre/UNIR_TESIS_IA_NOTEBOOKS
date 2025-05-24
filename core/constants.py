from pathlib import Path

ROOT = Path(__file__).parent.parent

OUTPUT_PARSER_PROMPT = """
The following JSON has an error:
{json}

The error that occurred while parsing the agent reasoning: {e}

Fix the JSON formatting, return only the fixed JSON as a code block
"""
