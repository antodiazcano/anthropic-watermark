"""Script to compare normal sampling with watermark sampling."""

from src.base_model import DummyLLM
from src.seed_watermark import SeedWatermarkModel
from src.tournament_watermark import TournamentWatermarkModel


def check(model: DummyLLM) -> None:
    """Checks if the watermarks are implemented correctly.

    Args:
        model: Model used to implement the watermark.
    """

    text_length = 50
    secret = "hello"
    context_length = 3
    query = "The garden hidden"

    plain_tokens = model.generate(query, text_length, context_length)
    watermarked_tokens = model.generate(
        query, text_length, context_length, secret=secret
    )
    watermarked_tokens[-1] = 0

    plain_score = model.detect(query, plain_tokens, context_length, secret)
    watermarked_score = model.detect(query, watermarked_tokens, context_length, secret)

    print(f"Plain text: {model.tokens_to_text(plain_tokens)}")
    print(f"\nWatermarked text: {model.tokens_to_text(watermarked_tokens)}")

    print(f"\nScore plain: {plain_score:.2f}")
    print(f"\nScore watermarked: {watermarked_score:.2f}")


def main() -> None:
    """Checks the seed and tournament watermark models."""

    check(SeedWatermarkModel())
    print("\n" * 3)
    check(TournamentWatermarkModel())


if __name__ == "__main__":
    main()
