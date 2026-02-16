import json
import re
from pathlib import Path

from google import genai

from scrapers.utils import clean_html

_client = None


def load_gemini_key(path: Path = Path("gemini_api_key.txt")) -> str:
    return path.read_text().strip()


def get_client(api_key: str | None = None) -> genai.Client:
    global _client
    if _client is None:
        key = api_key or load_gemini_key()
        _client = genai.Client(api_key=key)
    return _client


def configure_gemini(api_key: str | None = None) -> None:
    get_client(api_key)


def build_extraction_prompt(html_text: str) -> str:
    return f"""Extract the following fields from this professor's bio page. Return ONLY valid JSON with these fields:

- "name": full name (string)
- "title": academic title like "Associate Professor", "Professor", "Assistant Professor" (string or null)
- "department": department name (string or null)
- "email": email address (string or null)
- "phone": phone number (string or null)
- "expertise_raw": a 1-3 sentence summary of their research interests and expertise (string)
- "expertise_keywords": a list of 3-10 specific expertise keywords/phrases (list of strings)

If a field is not found, use null. For expertise_keywords, extract specific research topics, not generic terms.

Bio page content:
{html_text}"""


def parse_gemini_response(response_text: str) -> dict | None:
    text = response_text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_profile(html: str, model_name: str = "gemini-2.0-flash") -> dict | None:
    cleaned = clean_html(html)
    prompt = build_extraction_prompt(cleaned)
    client = get_client()
    response = client.models.generate_content(model=model_name, contents=prompt)
    return parse_gemini_response(response.text)
