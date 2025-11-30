import requests

class APIClient:

    def __init__(self, url="http://localhost:8000/generate_comment"):
        self.url = url

    def generate_comment(self, code: str, language: str = "python"):
        payload = {
            "code": code,
            "language": language
        }

        response = requests.post(self.url, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"API error: {response.text}")

        json_resp = response.json()
        return json_resp.get("comment", "")
