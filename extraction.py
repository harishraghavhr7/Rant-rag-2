import json
import requests
from typing import Dict, List


def extract_facts(text: str) -> Dict:
    """
    Use LLM to extract structured facts from conversation.
    
    Returns:
    {
        "foods": ["idli", "coffee"],
        "tasks": [
            {"text": "implement feature", "status": "completed"},
            {"text": "deploy", "status": "pending"}
        ]
    }
    """
    
    extraction_prompt = f"""
    Extract the following information from the conversation below.
    
    Return ONLY valid JSON (no markdown, no backticks).
    
    If no information is found for a category, use empty arrays.
    
    CONVERSATION:
    {text}
    
    INSTRUCTIONS:
    1. Extract all foods/drinks mentioned: "foods": ["item1", "item2", ...]
    2. Extract all tasks mentioned:
       - If user says "I completed X" or "finished X" or "done with X": status = "completed"
       - If user says "I need to X" or "I want to X" or "should X": status = "pending"
       - Default to "pending" if unclear
       "tasks": [
         {{"text": "task description", "status": "completed" or "pending"}},
         ...
       ]
    
    Return JSON format:
    {{
        "foods": [],
        "tasks": []
    }}
    """
    
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:1b",
                "prompt": extraction_prompt,
                "stream": False
            },
            timeout=120
        )
        
        response_text = res.json().get("response", "")
        
        # Try to parse JSON from response
        try:
            extracted = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response (in case LLM wraps it)
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
            else:
                extracted = {"foods": [], "tasks": []}
        
        # Validate structure
        if not isinstance(extracted.get("foods"), list):
            extracted["foods"] = []
        if not isinstance(extracted.get("tasks"), list):
            extracted["tasks"] = []
        
        # Validate tasks have required fields
        extracted["tasks"] = [
            t for t in extracted["tasks"]
            if isinstance(t, dict) and "text" in t
        ]
        
        return extracted
        
    except Exception as e:
        print(f"Error extracting facts: {e}")
        return {"foods": [], "tasks": []}