"""Claude as an optional second model. CLAUDE_MODEL=claude-sonnet-5-5 in .env (with
ANTHROPIC_API_KEY) makes the Gap Analyser and the pick of a company's own site use
Claude; without the line or the key, gpt-4o-mini is used as before."""
import json
import os
import re

from dotenv import load_dotenv


def claude_model():
    """The Claude model to use, or "" when Claude is not set up."""
    load_dotenv()
    model = os.environ.get("CLAUDE_MODEL", "")
    return model if model.startswith("claude") and os.environ.get("ANTHROPIC_API_KEY") else ""


def claude_json(prompt, system="", model=None):
    """Claude's reply to the prompt, read as one JSON object."""
    import anthropic  # only needed when Claude is used
    response = anthropic.Anthropic().messages.create(
        model=model or claude_model(), max_tokens=4000, **({"system": system} if system else {}),
        messages=[{"role": "user", "content": prompt + "\n\nReply with the JSON object only."}])
    text = "".join(block.text for block in response.content if block.type == "text")
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
