"""Tiny language model used for the demonstration."""

import json
import os
from abc import ABC, abstractmethod

import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr


class DummyLLM(ABC):
    """Class to simulate a simple LLM."""

    def __init__(self, secret: str) -> None:
        """Constructor of the class.

        Args:
            secret: Secret for the watermark.
        """

        self.vocabulary = [
            "The",
            "A",
            "Every",
            "quiet",
            "river",
            "garden",
            "carries",
            "shapes",
            "reveals",
            "new",
            "hidden",
            "distant",
            "today",
            "again",
            ".",
        ]
        self.tokens = list(range(len(self.vocabulary)))
        self.secret = secret

    def text_to_tokens(self, text: str) -> list[int]:
        """Transforms a text to its corresponding tokens.

        Args:
            text: Text to transform.

        Returns:
            Token corresponding to each word.

        Raises:
            ValueError: If a word does not belong to the vocabulary.
        """

        tokens = []

        for word in text.split():
            if word not in self.vocabulary:
                raise ValueError(f"'{word}' is not in the vocabulary")

            tokens.append(self.vocabulary.index(word))

        return tokens

    def tokens_to_text(self, tokens: list[int]) -> str:
        """Turn the toy model's token IDs into text.

        Args:
            tokens: List of generated tokens.

        Returns:
            Text that represent the tokens.
        """

        return " ".join(self.vocabulary[token] for token in tokens)

    def _forward(self) -> list[float]:
        """Obtains a probability for each word of the vocabulary.

        Returns:
            One probability for each word.
        """

        probs = range(1, len(self.vocabulary) + 1)
        normalized_probs = [p / sum(probs) for p in probs]

        return normalized_probs

    def _sample_next_token(self, probs: list[float]) -> int:
        """Obtains the next token.

        Args:
            probs: Probabilities for each token.

        Returns:
            Next token.
        """

        return np.random.choice(self.tokens, p=probs)

    @abstractmethod
    def _sample_next_token_watermark(self, probs: list[float]) -> int:
        """Obtains the next token.

        Args:
            probs: Probabilities for each token.

        Returns:
            Next token.
        """

    @staticmethod
    def _select_for_watermark(probs: list[float], threshold: float = 0.5) -> bool:
        """Decides if a token is selected for watermarking or not. The idea is to select
        a token with high entropy.

        Args:
            probs: Probabilities for each token.
            threshold: Threshold from which we predict high entropy.

        Returns:
            `True` if the token is selected, `False` otherwise.
        """

        entropy = sum(-p * np.log2(p) if p > 0 else 0 for p in probs)
        normalized_entropy = entropy / np.log2(len(probs))

        return normalized_entropy >= threshold

    def generate(self, length: int, watermark: bool = False) -> list[int]:
        """Generates text.

        Args:
            length: Length of the text.
            watermark: `True` to include watermark, `False` otherwise.

        Returns:
            Generated tokens.
        """

        output_tokens: list[int] = []

        while len(output_tokens) < length:
            probs = self._forward()

            if watermark and self._select_for_watermark(probs):
                next_token = self._sample_next_token_watermark(probs)
            else:
                next_token = self._sample_next_token(probs)

            output_tokens.append(next_token)

        return output_tokens

    def _select_for_detection(
        self, tokens: list[int], temperature: float = 0.5
    ) -> list[int]:
        """Selects the indexes of the tokens to inspect for detection.

        Args:
            tokens: Generated tokens.
            temperature: Temperature of the LLM.

        Returns:
            Indexes of the tokens to inspect.

        Raises:
            ValueError: If the API key is not provided.
            TypeError: If the response of the LLM is not correct.
        """

        text = self.tokens_to_text(tokens)
        indexed_tokens = [f"{i}: {t}" for i, t in enumerate(text.split())]
        prompt = (
            "Identify tokens that are likely to be variable (e.g., highly "
            "context-dependent, carrying low certainty, or open to multiple alternative"
            f" word choices).\nTokens: {indexed_tokens}\n\nReturn ONLY a JSON array of "
            "integers representing the selected indices, e.g. `[2, 5, 11]`."
        )

        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("You must provide an API key!")

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=temperature,
            api_key=SecretStr(api_key),
        )
        response = llm.invoke(prompt).content

        if not isinstance(response, str):
            raise TypeError("An error occurred when running the LLM!")

        return json.loads(response)

    @abstractmethod
    def detect(self, tokens: list[int]) -> float:
        """Calculates a watermark score for a sequence of tokens.

        Args:
            tokens: List of generated tokens.

        Returns:
            Watermark score between 0 and 1.
        """
