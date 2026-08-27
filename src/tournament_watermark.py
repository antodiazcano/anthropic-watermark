"""Tournament-based watermark sampling and detection."""

import hashlib
import hmac
from collections.abc import Callable

import numpy as np

from src.base_model import LanguageModel


class TournamentWatermarkModel(LanguageModel):
    """Language model using a three-round sampling tournament."""

    def _g1(self, token: int) -> int:
        """Returns the token's secret HMAC-SHA256 bit.

        Args:
            token: Token ID to compute the secret bit for.

        Returns:
            Secret bit as an integer (0 or 1).
        """

        return hmac.digest(self.secret.encode(), str(token).encode(), "sha256")[0] % 2

    def _g2(self, token: int) -> int:
        """Returns the token's secret SHA3-256 bit.

        Args:
            token: Token ID to compute the secret bit for.

        Returns:
            Secret bit as an integer (0 or 1).
        """

        data = f"{self.secret}:{token}".encode()
        return hashlib.sha3_256(data).digest()[0] & 1

    def _g3(self, token: int) -> int:
        """Returns the token's secret keyed-BLAKE2b bit.

        Args:
            token: Token ID to compute the secret bit for.

        Returns:
            Secret bit as an integer (0 or 1).
        """

        hasher = hashlib.blake2b(key=self.secret.encode(), digest_size=1)
        hasher.update(token.to_bytes(4, byteorder="big", signed=True))
        return hasher.digest()[0] % 2

    def _get_gs(self) -> list[Callable[[int], int]]:
        """Returns the secret bit functions.

        Returns:
            List of functions that compute secret bits for tokens.
        """

        return [self._g1, self._g2, self._g3]

    def _tournament(
        self, candidates: list[int], signatures: list[list[int]], current_bit: int = 0
    ) -> int:
        """Compare candidate pairs until one token remains.

        Args:
            candidates: List of candidate token IDs.
            signatures: List of secret bit signatures for each candidate.
            current_bit: Index of the current secret bit to compare.

        Returns:
            The winning token ID after the tournament.
        """

        if len(candidates) == 1:
            return candidates[0]

        new_candidates: list[int] = []
        new_signatures: list[list[int]] = []

        for index in range(0, len(candidates), 2):
            # Select corresponding bits from the signatures of the two candidates
            first_bit = signatures[index][current_bit]
            second_bit = signatures[index + 1][current_bit]
            # Determine the winner
            if first_bit > second_bit:
                winner = index
            elif second_bit > first_bit:
                winner = index + 1
            else:
                winner = index + int(np.random.choice([0, 1]))
            # Add winner and signature for the next round
            new_candidates.append(candidates[winner])
            new_signatures.append(signatures[winner])

        return self._tournament(new_candidates, new_signatures, current_bit + 1)

    def _sample_next_token_watermark(self, probs: list[float]) -> int:
        """Samples 2**m candidates and returns the tournament winner.

        Args:
            probs: Probabilities for the next token.

        Returns:
            Sampled token ID after the tournament.
        """

        g_functions = self._get_gs()
        m = len(g_functions)

        candidates: list[int] = []
        signatures: list[list[int]] = []

        for _ in range(2**m):
            candidate = int(np.random.choice(self.tokens, p=probs))
            candidates.append(candidate)
            signatures.append([g(candidate) for g in g_functions])

        return self._tournament(candidates, signatures)

    def detect_tokens(self, tokens: list[int]) -> float:
        """Computes a watermark score from token IDs.

        Args:
            tokens: Token IDs to score.

        Returns:
            Watermark score as a float.
        """

        # The set is to avoid duplicated indexes, just in case
        indexes_to_inspect = set(self._select_for_detection(tokens))

        g_functions = self._get_gs()
        g_bits = sum(
            sum(g(tokens[index]) for g in g_functions) for index in indexes_to_inspect
        )

        # g_bits = 0
        # for token in tokens_to_inspect:
        #     g_bits += sum(g(token) for g in g_functions)

        return g_bits / (len(indexes_to_inspect) * len(g_functions))
