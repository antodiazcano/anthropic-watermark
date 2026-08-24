"""Script to implement the seed-based watermark."""

import hashlib
import hmac
import struct

import numpy as np

from src.base_model import DummyLLM


class SeedWatermarkModel(DummyLLM):
    """Class to simulate a simple LLM with watermark based on the seed."""

    def _sample_next_token_watermark(self, context: list[int], secret: str) -> int:
        """Obtains the next token.

        Args:
            context: Previous tokens.
            secret: Secret key.

        Returns:
            Next token.
        """

        probs = self._forward(context)

        # Compute HMAC-SHA256 and convert first `n_bytes` bytes to an integer of
        # 8 * `n_bytes` bits. The choice of 4 is because of the limit of numpy seed.
        context_str = ",".join(map(str, context))
        n_bytes = 4
        hmac_digest = hmac.new(
            key=secret.encode("utf-8"),
            msg=context_str.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        seed = struct.unpack(">I", hmac_digest[:n_bytes])[0]
        np.random.seed(seed)

        return np.random.choice(self.tokens, p=probs)

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

        watermark_tokens = self.generate(
            query, len(tokens), context_length=context_length, secret=secret
        )
        equal_tokens = [
            1 if token == watermark_token else 0
            for token, watermark_token in zip(tokens, watermark_tokens)
        ]

        return sum(equal_tokens) / len(equal_tokens)
