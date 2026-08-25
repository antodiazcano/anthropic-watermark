"""Shared language-model code for the watermark examples."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LanguageModel(ABC):
    """Small causal language model with optional watermark sampling."""

    def __init__(
        self,
        secret: str,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        top_k: int = 50,
        entropy_threshold: float = 0.5,
    ) -> None:
        """Load the tokenizer and model used by generation and detection.

        Args:
            secret: Secret key used by the watermark.
            model_name: Hugging Face model to load.
            top_k: Number of likely tokens kept before sampling.
            entropy_threshold: Minimum normalized entropy for watermarking.
        """

        self.secret = secret
        self.top_k = top_k
        self.entropy_threshold = entropy_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.tokens = list(range(self.model.config.vocab_size))

    def text_to_tokens(self, text: str) -> list[int]:
        """Convert text into token IDs."""

        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def tokens_to_text(self, tokens: list[int]) -> str:
        """Convert token IDs into text."""

        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _forward(self, context: list[int]) -> list[float]:
        """Return the top-k next-token distribution for a context."""

        input_ids = torch.tensor([context], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            logits = self.model(input_ids=input_ids).logits[0, -1]

        top_k = min(self.top_k, len(logits))
        top_logits, top_tokens = torch.topk(logits, top_k)
        top_probs = torch.softmax(top_logits, dim=0)
        probs = torch.zeros_like(logits)
        probs[top_tokens] = top_probs

        return probs.cpu().tolist()

    def _sample_next_token(self, probs: list[float]) -> int:
        """Sample a token from a model distribution."""

        return int(np.random.choice(self.tokens, p=probs))

    @abstractmethod
    def _sample_next_token_watermark(self, probs: list[float]) -> int:
        """Sample a token using the watermark method."""

    @staticmethod
    def _select_for_watermark(probs: list[float], threshold: float) -> bool:
        """Return whether a distribution has enough entropy to watermark."""

        positive_probs = [prob for prob in probs if prob > 0]
        if len(positive_probs) < 2:
            return False

        entropy = -sum(prob * np.log2(prob) for prob in positive_probs)
        normalized_entropy = entropy / np.log2(len(positive_probs))

        return normalized_entropy >= threshold

    def generate(self, prompt: str, length: int, watermark: bool = False) -> list[int]:
        """Generate continuation tokens from a prompt.

        Args:
            prompt: Text that starts the generation.
            length: Number of new tokens to generate.
            watermark: Whether to use watermark sampling.

        Returns:
            Newly generated token IDs, without the prompt tokens.

        Raises:
            ValueError: If the prompt produces no tokens.
        """

        context = self.text_to_tokens(prompt)
        if not context:
            raise ValueError("The prompt must not be empty")

        output_tokens: list[int] = []

        for _ in range(length):
            probs = self._forward(context)

            if watermark and self._select_for_watermark(probs, self.entropy_threshold):
                next_token = self._sample_next_token_watermark(probs)
            else:
                next_token = self._sample_next_token(probs)

            output_tokens.append(next_token)
            context.append(next_token)

        return output_tokens

    @abstractmethod
    def detect(self, prompt: str, tokens: list[int]) -> float:
        """Calculate a watermark score for continuation tokens."""
