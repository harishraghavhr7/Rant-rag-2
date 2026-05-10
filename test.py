import requests

res = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "gemma3:1b",
        "prompt": "hello",
        "stream": False
    }
)

print(res.status_code)
print(res.text)