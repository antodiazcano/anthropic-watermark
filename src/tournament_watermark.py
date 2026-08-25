"""Script to implement the tournament-based watermark."""

import hashlib
import hmac
from collections.abc import Callable

import numpy as np

from src.base_model import DummyLLM


class TournamentWatermarkModel(DummyLLM):
    """Class to simulate a simple LLM with watermark based on the tournament."""

    def _g1(self, token: int) -> int:
        """G1 function.

        Args:
            token: Token to which the function is applied.

        Returns:
            0 or 1 signature.
        """

        return hmac.digest(self.secret.encode(), str(token).encode(), "sha256")[0] % 2

    def _g2(self, token: int) -> int:
        """G2 function.

        Args:
            token: Token to which the function is applied.

        Returns:
            0 or 1 signature.
        """

        data = f"{self.secret}:{token}".encode()
        digest = hashlib.sha3_256(data).digest()
        return digest[0] & 1

    def _g3(self, token: int) -> int:
        """G3 function.

        Args:
            token: Token to which the function is applied.

        Returns:
            0 or 1 signature.
        """

        hasher = hashlib.blake2b(key=self.secret.encode("utf-8"), digest_size=1)
        hasher.update(int(token).to_bytes(4, byteorder="big", signed=True))
        return hasher.digest()[0] % 2

    def _get_gs(self) -> list[Callable[[int], int]]:
        """Obtains all the g-like functions.

        Returns:
            All g-functions.
        """

        return [self._g1, self._g2, self._g3]

    def _tournament(
        self, candidates: list[int], signatures: list[list[int]], current_bit: int = 0
    ) -> int:
        """Simulates a tournament to select the next token.

        Args:
            candidates: Token candidates.
            signatures: Signature of each candidate.
            current_bit: Current bit of the tournament.

        Returns:
            Winner token.
        """

        if len(candidates) == 1:
            return candidates[0]

        new_candidates = []
        new_signatures = []

        for i in range(0, len(candidates), 2):
            # Select two opponents
            temp_candidates = candidates[i : i + 2]
            # Select the corresponding bit of its signatures
            temp_signatures = signatures[i : i + 2]
            bit_first_candidate = temp_signatures[0][current_bit]
            bit_second_candidate = temp_signatures[1][current_bit]
            # Decide the winner
            if bit_first_candidate == 1 and bit_second_candidate == 0:
                winner = 0
            elif bit_first_candidate == 0 and bit_second_candidate == 1:
                winner = 1
            else:
                winner = np.random.choice([0, 1])
            # Pass winner and its signature to the next round
            new_candidates.append(temp_candidates[winner])
            new_signatures.append(temp_signatures[winner])

        return self._tournament(new_candidates, new_signatures, current_bit + 1)

    def _sample_next_token_watermark(self, probs: list[float]) -> int:
        """Obtains the next token. It samples candidates and their signatures and runs
        the tournament.

        Args:
            probs: Probabilities for each token.

        Returns:
            Next token.
        """

        g_functions = self._get_gs()
        m = len(g_functions)

        candidates = []
        signatures = []
        for _ in range(2**m):
            candidate_token = np.random.choice(self.tokens, p=probs)
            candidates.append(candidate_token)
            signatures.append([g(candidate_token) for g in g_functions])

        return self._tournament(candidates, signatures)

    def detect(self, tokens: list[int]) -> float:
        """Calculates a watermark score for a sequence of tokens.

        Args:
            tokens: List of generated tokens.

        Returns:
            Watermark score between 0 and 1.
        """

        g_functions = self._get_gs()
        total_bits = 0
        indexes_to_inspect = self._select_for_detection(tokens)

        for index_to_inspect in indexes_to_inspect:
            token_to_inspect = tokens[index_to_inspect]
            for g in g_functions:
                total_bits += g(token_to_inspect)

        return total_bits / (len(indexes_to_inspect) * len(g_functions))
