"""Tiny language model used for the demonstration."""

from abc import ABC, abstractmethod

import numpy as np


class DummyLLM(ABC):
    """Class to simulate a simple LLM."""

    def __init__(self) -> None:
        """Constructor of the class."""

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

    def _initial_context(self, query: str, context_length: int) -> list[int]:
        """Create a fixed-length context from the end of the query."""

        if context_length <= 0:
            raise ValueError("context_length must be greater than zero")

        query_tokens = self.text_to_tokens(query)
        query_context = query_tokens[-context_length:]
        padding = [0] * (context_length - len(query_context))

        return padding + query_context

    def tokens_to_text(self, tokens: list[int]) -> str:
        """Turn the toy model's token IDs into text.

        Args:
            tokens: List of generated tokens.

        Returns:
            Text that represent the tokens.
        """

        return " ".join(self.vocabulary[token] for token in tokens)

    def _forward(self, context: list[int]) -> list[float]:
        """Obtains a probability for each word of the vocabulary.

        Args:
            context: Previous tokens.

        Returns:
            One probability for each word.
        """

        probs = range(len(self.vocabulary))
        normalized_probs = [p / sum(probs) for p in probs]

        return normalized_probs

    def _sample_next_token(self, context: list[int]) -> int:
        """Obtains the next token.

        Args:
            context: Previous tokens.

        Returns:
            Next token.
        """

        probs = self._forward(context)
        return np.random.choice(self.tokens, p=probs)

    @abstractmethod
    def _sample_next_token_watermark(self, context: list[int], secret: str) -> int:
        """Obtains the next token.

        Args:
            context: Previous tokens.
            secret: Secret key.

        Returns:
            Next token.
        """

    def generate(
        self, query: str, length: int, context_length: int, *, secret: str | None = None
    ) -> list[int]:
        """Generate plain text, or watermarked text when a secret is supplied.

        Args:
            query: Query of the user.
            length: Length of the text.
            context_length: Length of the context used to generate the deterministic
                seed (if watermark is used).
            secret: Secret key used to generate the deterministic seed (if watermark is
                used).

        Returns:
            Generated text.
        """

        output_tokens: list[int] = []

        context = self._initial_context(query, context_length)

        while len(output_tokens) < length:
            temp_context = context[-context_length:]

            if secret is not None:
                next_token = self._sample_next_token_watermark(temp_context, secret)
            else:
                next_token = self._sample_next_token(temp_context)

            output_tokens.append(next_token)
            context.append(next_token)

        return output_tokens

    @abstractmethod
    def detect(
        self, query: str, tokens: list[int], context_length: int, secret: str
    ) -> float:
        """Calculates a watermark score for a sequence of tokens.

        Args:
            query: Query of the user.
            tokens: List of generated tokens.
            context_length: Length of the context used to generate the deterministic
                seed.
            secret: Secret key used to generate the deterministic seed.

        Returns:
            Watermark score between 0 and 1.
        """
