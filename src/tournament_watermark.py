"""Script to implement the tournament-based watermark."""

import hashlib
import hmac
from collections.abc import Callable

import numpy as np

from src.base_model import DummyLLM


class TournamentWatermarkModel(DummyLLM):
    """Class to simulate a simple LLM with watermark based on the tournament."""

    @staticmethod
    def _g1(context: list[int], token: int, secret: str) -> int:
        """G1 function.

        Args:
            context: Previous tokens.
            token: Token to which the function is applied.
            secret: Secret key.

        Returns:
            0 or 1 signature.
        """

        values = context + [token]
        message = ",".join(str(value) for value in values).encode()
        return hmac.digest(secret.encode(), message, "sha256")[0] % 2

    @staticmethod
    def _g2(context: list[int], token: int, secret: str) -> int:
        """G2 function.

        Args:
            context: Previous tokens.
            token: Token to which the function is applied.
            secret: Secret key.

        Returns:
            0 or 1 signature.
        """

        data = f"{secret}:{context}:{token}".encode()
        digest = hashlib.sha3_256(data).digest()
        return digest[0] & 1

    @staticmethod
    def _g3(context: list[int], token: int, secret: str) -> int:
        """G3 function.

        Args:
            context: Previous tokens.
            token: Token to which the function is applied.
            secret: Secret key.

        Returns:
            0 or 1 signature.
        """

        hasher = hashlib.blake2b(key=secret.encode("utf-8"), digest_size=1)

        for val in context + [token]:
            hasher.update(int(val).to_bytes(4, byteorder="big", signed=True))

        return hasher.digest()[0] % 2

    def _get_gs(self) -> list[Callable[[list[int], int, str], int]]:
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
            # Select its signatures
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
            # Pass to next round
            new_candidates.append(temp_candidates[winner])
            new_signatures.append(temp_signatures[winner])

        return self._tournament(new_candidates, new_signatures, current_bit + 1)

    def _sample_next_token_watermark(self, context: list[int], secret: str) -> int:
        """Obtains the next token.

        Args:
            context: Previous tokens.
            secret: Secret key (if provided, then watermark is applied).

        Returns:
            Next token.
        """

        g_functions = self._get_gs()
        m = len(g_functions)
        probs = self._forward(context)

        # Sample candidates and its signatures
        candidates = []
        signatures = []
        for _ in range(2**m):
            candidate_token = np.random.choice(self.tokens, p=probs)
            candidates.append(candidate_token)
            signatures.append(
                [g(context, candidate_token, secret) for g in g_functions]
            )

        return self._tournament(candidates, signatures)

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

        g_functions = self._get_gs()

        context = self._initial_context(query, context_length)

        total_bits = 0
        for token in tokens:
            temp_context = context[-context_length:]
            for g in g_functions:
                total_bits += g(temp_context, token, secret)
            context.append(token)

        return total_bits / (len(g_functions) * len(tokens))
