"""Shared language-model code for the watermark examples."""

import json
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr
from transformers import AutoModelForCausalLM, AutoTokenizer


class LanguageModel(ABC):
    """Small causal language model with optional watermark sampling."""

    def __init__(
        self, secret: str, model_name: str = "HuggingFaceTB/SmolLM2-135M"
    ) -> None:
        """Load the tokenizer and model used by generation and detection.

        Args:
            secret: Secret key used by the watermark.
            model_name: Hugging Face model to load.
        """

        self.secret = secret
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=self.device
        ).eval()
        self.tokens = list(range(self.model.config.vocab_size))

    def text_to_tokens(self, text: str) -> list[int]:
        """Convert text into token IDs.

        Args:
            text: Text to convert into token IDs.

        Returns:
            Token IDs corresponding to the input text.
        """

        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def tokens_to_text(self, tokens: list[int]) -> str:
        """Convert token IDs into text.

        Args:
            tokens: Token IDs to convert into text.

        Returns:
            Text corresponding to the input token IDs.

        Raises:
            TypeError: If the type of the decoded text is incorrect.
        """

        text = self.tokenizer.decode(tokens, skip_special_tokens=True)

        if not isinstance(text, str):
            raise TypeError("Text was not converted to string!")

        return text

    def _forward(self, context: list[int]) -> list[float]:
        """Return the next-token distribution for a context.

        Args:
            context: List of token IDs representing the context.

        Returns:
            Probabilities for the next token.
        """

        input_ids = torch.tensor([context], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            logits = self.model(input_ids=input_ids).logits[0, -1]

        top_k = 30
        top_logits, top_tokens = torch.topk(logits, top_k)
        # This strange dtype changes is for the probs to sum up to 1
        top_probs = torch.softmax(top_logits.double(), dim=0)
        probs = torch.zeros_like(logits, dtype=torch.float64)
        probs[top_tokens] = top_probs

        return probs.cpu().tolist()

    def _sample_next_token(self, probs: list[float]) -> int:
        """Sample a token from a model distribution.

        Args:
            probs: Probabilities for the next token.

        Returns:
            Sampled token ID.
        """

        return int(np.random.choice(self.tokens, p=probs))

    @abstractmethod
    def _sample_next_token_watermark(self, probs: list[float]) -> int:
        """Sample a token using the watermark method.

        Args:
            probs: Probabilities for the next token.

        Returns:
            Sampled token ID.
        """

    @staticmethod
    def _select_for_watermark(probs: list[float], threshold: float = 0.5) -> bool:
        """Decides whether a distribution has enough entropy to watermark.

        Args:
            probs: Probabilities for the next token.
            threshold: Minimum normalized entropy to watermark.

        Returns:
            `True` if the distribution has enough entropy to watermark, `False`
            otherwise.
        """

        positive_probs = [p for p in probs if p > 0]
        entropy = -sum(p * np.log2(p) for p in positive_probs)
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
        """

        context = self.text_to_tokens(prompt)
        output_tokens: list[int] = []

        for _ in range(length):
            probs = self._forward(context)

            if watermark and self._select_for_watermark(probs):
                next_token = self._sample_next_token_watermark(probs)
            else:
                next_token = self._sample_next_token(probs)

            output_tokens.append(next_token)
            context.append(next_token)

        return output_tokens

    def _select_for_detection(self, tokens: list[int]) -> list[int]:
        """Selects the indexes of the tokens to inspect for detection.

        Args:
            tokens: Generated tokens.

        Returns:
            Indexes of the tokens to inspect.

        Raises:
            ValueError: If the API key is not provided.
            TypeError: If the response of the LLM is not correct.
        """

        indexed_tokens = [
            f"{index}: {self.tokenizer.convert_ids_to_tokens(token)!r}"
            for index, token in enumerate(tokens)
        ]  # !r is for showing ok special characters
        prompt = (
            "Identify tokens that are likely to be variable (e.g., highly "
            "context-dependent, carrying low certainty, or open to multiple alternative"
            f" word choices). Don't include the 0 index. \nTokens: {indexed_tokens}\n\n"
            "Return ONLY a JSON array of integers representing the selected indices, "
            "e.g. `[2, 5, 11]`."
        )

        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("You must provide an API key!")

        llm = ChatGroq(
            model="openai/gpt-oss-120b", temperature=0.5, api_key=SecretStr(api_key)
        )
        response = llm.invoke(prompt).content
        if not isinstance(response, str):
            raise TypeError("An error occurred when running the LLM!")
        indexes = json.loads(response)

        return [index for index in indexes if 0 < index < len(tokens)]

    def detect(self, text: str) -> float:
        """Computes a watermark score for a text.

        Args:
            text: Text to score.

        Returns:
            Watermark score as a float.
        """

        return self.detect_tokens(self.text_to_tokens(text))

    @abstractmethod
    def detect_tokens(self, tokens: list[int]) -> float:
        """Computes a watermark score from token IDs.

        Args:
            tokens: Token IDs to score.

        Returns:
            Watermark score as a float.
        """
