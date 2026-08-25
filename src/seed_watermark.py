"""Script to implement the seed-based watermark."""

import hashlib
import hmac
import struct

import numpy as np

from src.base_model import DummyLLM


class SeedWatermarkModel(DummyLLM):
    """Class to simulate a simple LLM with watermark based on the seed."""

    def _fix_seed(self) -> None:
        """Compute HMAC-SHA256 and convert first `n_bytes` bytes to an integer of
        8 * `n_bytes` bits. The choice of 4 is because of the limit of numpy seed.
        """

        n_bytes = 4
        hmac_digest = hmac.new(
            key=self.secret.encode("utf-8"),
            msg=self.secret.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        seed = struct.unpack(">I", hmac_digest[:n_bytes])[0]
        np.random.seed(seed)

    def _sample_next_token_watermark(self, probs: list[float]) -> int:
        """Obtains the next token.

        Args:
            probs: Probabilities for each token.

        Returns:
            Next token.
        """

        self._fix_seed()
        return np.random.choice(self.tokens, p=probs)

    def detect(self, tokens: list[int]) -> float:
        """Calculates a watermark score for a sequence of tokens.

        Args:
            tokens: List of generated tokens.

        Returns:
            Watermark score between 0 and 1.
        """

        equal = []
        indexes_to_inspect = self._select_for_detection(tokens)

        for index_to_inspect in indexes_to_inspect:
            probs = self._forward()
            token_to_inspect = tokens[index_to_inspect]
            if token_to_inspect == self._sample_next_token_watermark(probs):
                equal.append(1)
            else:
                equal.append(0)

        return sum(equal) / len(equal) if equal else 0.0
