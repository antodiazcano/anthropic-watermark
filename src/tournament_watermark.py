"""Tournament-based watermark sampling and detection."""

import hashlib
import hmac
from collections.abc import Callable

import numpy as np

from src.base_model import LanguageModel


class TournamentWatermarkModel(LanguageModel):
    """Language model using a three-round sampling tournament."""

    def _g1(self, token: int) -> int:
        """Return the token's secret HMAC-SHA256 bit."""

        return hmac.digest(self.secret.encode(), str(token).encode(), "sha256")[0] % 2

    def _g2(self, token: int) -> int:
        """Return the token's secret SHA3-256 bit."""

        data = f"{self.secret}:{token}".encode()
        return hashlib.sha3_256(data).digest()[0] & 1

    def _g3(self, token: int) -> int:
        """Return the token's secret keyed-BLAKE2b bit."""

        hasher = hashlib.blake2b(key=self.secret.encode(), digest_size=1)
        hasher.update(token.to_bytes(4, byteorder="big", signed=True))
        return hasher.digest()[0] % 2

    def _get_gs(self) -> list[Callable[[int], int]]:
        """Return the three secret bit functions."""

        return [self._g1, self._g2, self._g3]

    def _tournament(
        self, candidates: list[int], signatures: list[list[int]], current_bit: int = 0
    ) -> int:
        """Compare candidate pairs until one token remains."""

        if len(candidates) == 1:
            return candidates[0]

        new_candidates: list[int] = []
        new_signatures: list[list[int]] = []

        for index in range(0, len(candidates), 2):
            first_bit = signatures[index][current_bit]
            second_bit = signatures[index + 1][current_bit]

            if first_bit > second_bit:
                winner = index
            elif second_bit > first_bit:
                winner = index + 1
            else:
                winner = index + int(np.random.choice([0, 1]))

            new_candidates.append(candidates[winner])
            new_signatures.append(signatures[winner])

        return self._tournament(new_candidates, new_signatures, current_bit + 1)

    def _sample_next_token_watermark(self, probs: list[float]) -> int:
        """Sample eight candidates and return the tournament winner."""

        g_functions = self._get_gs()
        candidates: list[int] = []
        signatures: list[list[int]] = []

        for _ in range(2 ** len(g_functions)):
            candidate = int(np.random.choice(self.tokens, p=probs))
            candidates.append(candidate)
            signatures.append([g(candidate) for g in g_functions])

        return self._tournament(candidates, signatures)

    def detect(self, prompt: str, tokens: list[int]) -> float:
        """Return the mean secret bit at selected token positions."""

        context = self.text_to_tokens(prompt)
        g_functions = self._get_gs()
        total_bits = 0
        selected_tokens = 0

        for token in tokens:
            probs = self._forward(context)

            if self._select_for_watermark(probs, self.entropy_threshold):
                total_bits += sum(g(token) for g in g_functions)
                selected_tokens += 1

            context.append(token)

        total_selected_bits = selected_tokens * len(g_functions)
        return total_bits / total_selected_bits if total_selected_bits else 0.0
