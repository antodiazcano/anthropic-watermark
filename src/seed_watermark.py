"""Seed-based watermark sampling and detection."""

import hashlib
import hmac
import struct

import numpy as np

from src.base_model import LanguageModel


class SeedWatermarkModel(LanguageModel):
    """Language model using secret-seeded sampling."""

    def _fix_seed(self) -> None:
        """Set NumPy's seed from the first four bytes of a secret HMAC."""

        digest = hmac.new(
            key=self.secret.encode("utf-8"),
            msg=self.secret.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        seed = struct.unpack(">I", digest[:4])[0]
        np.random.seed(seed)

    def _sample_next_token_watermark(self, probs: list[float]) -> int:
        """Sample a token using the watermark method.

        Args:
            probs: Probabilities for the next token.

        Returns:
            Sampled token ID.
        """

        self._fix_seed()
        return int(np.random.choice(self.tokens, p=probs))

    def detect(self, text: str) -> float:
        """Computes a watermark score for a text.

        Args:
            text: Text to score.

        Returns:
            Watermark score as a float.
        """

        tokens = self.text_to_tokens(text)
        indexes_to_inspect = self._select_for_detection(tokens)
        matches: list[int] = []
        context: list[int] = []

        for index, token in enumerate(tokens):
            if index in indexes_to_inspect:
                probs = self._forward(context)
                expected_token = self._sample_next_token_watermark(probs)
                if token == expected_token:
                    matches.append(1)
                else:
                    matches.append(0)

            context.append(token)

        return sum(matches) / len(matches)
