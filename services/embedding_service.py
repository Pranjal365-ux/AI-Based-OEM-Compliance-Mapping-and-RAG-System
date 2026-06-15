import requests


class EmbeddingService:
    def __init__(
        self,
        base_url: str = "http://192.168.2.123:11434",
        model: str = "bge-m3",
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()["embeddings"]