import os, sys, numpy as np

CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from step_2_generate_drum_ssm import (
    get_extracted_full_ssm,
    get_extracted_triangle_ssm,
)

def test_get_extracted_full_ssm_crops_square():
    S = np.arange(10*10, dtype=np.float32).reshape(10, 10)
    out = get_extracted_full_ssm(S, 6)
    assert out.shape == (6, 6)
    # top-left 6x6 should match
    np.testing.assert_array_equal(out, S[:6, :6])

def test_get_extracted_triangle_ssm_is_symmetric():
    # Make an asymmetric 6x6
    rng = np.random.default_rng(0)
    S = rng.normal(size=(6, 6)).astype(np.float32)
    T = get_extracted_triangle_ssm(S, 6)
    assert T.shape == (6, 6)
    # Symmetry check
    np.testing.assert_allclose(T, T.T, rtol=0, atol=1e-6)
    # Diagonal preserved from the upper triangle (includes diag)
    np.testing.assert_allclose(np.diag(T), np.diag(np.triu(S)))
