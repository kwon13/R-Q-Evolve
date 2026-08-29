from difflib import SequenceMatcher

import pytest

from rq_evolve.similarity import (
    SimilarityTimeout,
    _SimilarityClient,
    matched_size,
    sequence_ratio,
)


def test_similarity_worker_preserves_historical_sequence_matcher_metric():
    left = "Let n = N. Count the divisors of n."
    right = "Let n = N be positive. Count the divisors of n."
    expected = SequenceMatcher(None, left, right, autojunk=False)

    assert sequence_ratio(left, right) == pytest.approx(expected.ratio())
    assert matched_size(left, right) == sum(
        block.size for block in expected.get_matching_blocks()
    )


def test_pathological_similarity_is_killed_and_worker_recovers():
    client = _SimilarityClient()
    try:
        # Alternating, shifted sequences are a quadratic adversarial case when
        # autojunk is deliberately disabled by the archive's historical gate.
        left = "ab" * 10_000
        right = "ba" * 10_000
        with pytest.raises(SimilarityTimeout):
            client.run("ratio", left, right, autojunk=False, timeout=0.01)

        assert client.run(
            "ratio", "abc", "abc", autojunk=False, timeout=2.0
        ) == pytest.approx(1.0)
    finally:
        client.close()
