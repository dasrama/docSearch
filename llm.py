class LocalLLM:
    def __init__(self, model_path: str = None, n_ctx: int = 2048):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self._client = None
        try:
            from llama_cpp import Llama
            if model_path:
                self._client = Llama(model_path=model_path, n_ctx=n_ctx)
        except Exception:
            self._client = None

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        if self._client is not None:
            resp = self._client.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            return resp.get("choices", [{}])[0].get("text", "").strip()
        # fallback simple echo/generative placeholder
        # In production, install `llama-cpp-python` and provide `model_path`.
        return "[LOCAL_LLM_FALLBACK]\n" + prompt[:200]
