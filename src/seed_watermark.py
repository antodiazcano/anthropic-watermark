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
        """Sample after resetting NumPy to the secret-derived seed."""

        self._fix_seed()
        return int(np.random.choice(self.tokens, p=probs))

    def detect(self, prompt: str, tokens: list[int]) -> float:
        """Return the fraction of selected tokens reproduced by the secret."""

        context = self.text_to_tokens(prompt)
        matches: list[int] = []

        for token in tokens:
            probs = self._forward(context)

            if self._select_for_watermark(probs, self.entropy_threshold):
                expected_token = self._sample_next_token_watermark(probs)
                matches.append(int(token == expected_token))

            context.append(token)

        return sum(matches) / len(matches) if matches else 0.0
