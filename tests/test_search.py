import numpy as np
import pytest
from llambda.search import normalize, similarity_argsort


def test_normalize():
    normalized = normalize(np.array([1, 2, 3]))
    np.testing.assert_almost_equal((normalized ** 2).sum(), 1)


@pytest.fixture(scope="module")
def vectors():
    return np.array([
        [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]],
        [[0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]],
        [[0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]],
    ])



@pytest.mark.parametrize(
    "query, expected",
    [
        (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]), [0, 1, 2]),
        (np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]), [1, 0, 2]),
        (np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]), [2, 0, 1]),
        (np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]), [0, 1, 2]),
        (np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]), [1, 0, 2]),
        (np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]), [2, 0, 1]),
        (np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]), [0, 1, 2]),
        (np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]), [1, 0, 2]),
        (np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]), [2, 0, 1]),
        (np.array([1, 0, 0.5, 0, 0, 0, 0, 0, 0]), [0, 2, 1]),
        (np.array([0, 1, 0, 0, 0, 0.5, 0, 0, 0]), [1, 2, 0]),
        (np.array([0, 0, 1, 0, 0, 0, 0, 0.5, 0]), [2, 1, 0]),
        (np.array([0, 0, 0, 1, 0, 0, 0, 0, 0.5]), [0, 2, 1]),
        (np.array([0, 0, 0.8, 0, 1, 0.8, 0, 0, 0.8]), [1, 2, 0]),
        (np.array([0, 0.8, 0, 0, 0.8, 1, 0, 0.8, 0.5]), [2, 1, 0]),
        (np.array([0.5, 0, 0.8, 0, 0, 0, 1, 0, 0]), [0, 2, 1]),
    ]
)
def test_similarity_argsort(query, expected, vectors):
    assert similarity_argsort(query, vectors) == expected
