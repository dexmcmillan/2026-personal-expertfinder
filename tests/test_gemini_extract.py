import json

from gemini_extract import build_extraction_prompt, parse_gemini_response


def test_build_extraction_prompt_includes_html():
    prompt = build_extraction_prompt("Dr. Smith studies AI ethics.")
    assert "Dr. Smith studies AI ethics." in prompt
    assert "name" in prompt.lower()
    assert "expertise" in prompt.lower()


def test_parse_gemini_response_valid_json():
    response_text = json.dumps({
        "name": "Jane Smith",
        "title": "Associate Professor",
        "department": "Computer Science",
        "email": "jsmith@uwaterloo.ca",
        "phone": None,
        "expertise_raw": "Machine learning and computer vision research.",
        "expertise_keywords": ["machine learning", "computer vision"],
    })
    result = parse_gemini_response(response_text)
    assert result["name"] == "Jane Smith"
    assert result["expertise_keywords"] == ["machine learning", "computer vision"]


def test_parse_gemini_response_json_in_markdown():
    response_text = '```json\n{"name": "Jane Smith", "title": "Prof", "department": "CS", "email": null, "phone": null, "expertise_raw": "AI.", "expertise_keywords": ["AI"]}\n```'
    result = parse_gemini_response(response_text)
    assert result["name"] == "Jane Smith"


def test_parse_gemini_response_invalid_returns_none():
    result = parse_gemini_response("This is not JSON at all.")
    assert result is None
