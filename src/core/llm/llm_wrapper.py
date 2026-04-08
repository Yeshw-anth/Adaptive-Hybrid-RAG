class LLMWrapper:
    def __init__(self, client):
        self.client = client

    def generate(self, prompt, **kwargs):
        return self.client.chat(prompt=prompt, **kwargs)