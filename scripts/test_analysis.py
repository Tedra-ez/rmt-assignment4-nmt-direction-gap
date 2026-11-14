"""Unit tests of the statistical functions on small invented data.

Checks: DeltaGap formula, bootstrap pairing, permutation p-values with the +1
correction, Holm correction, and leave-one-out results.
Run: python -m unittest discover -s scripts -p "test_analysis.py"
"""
import unittest

import numpy as np
import pandas as pd

from analyze import MODEL_IDS, additional_analyses, bootstrap, group_outcomes, holm_adjust, permutation_test


class StatisticalKernelTests(unittest.TestCase):
    def setUp(self):
        # Six invented languages, two invented models, two directions.
        base = np.array([[30., 28.], [34., 30.], [29., 28.],
                         [20., 12.], [18., 9.], [22., 15.]])
        self.cube = np.stack((base, 2 * base + 3), axis=1)
        self.high = np.array([True, True, True, False, False, False])

    def test_delta_gap_identity_and_model_pairing(self):
        outcome = group_outcomes(self.cube, self.high)
        d = self.cube[:, :, 0] - self.cube[:, :, 1]
        np.testing.assert_allclose(outcome[:, -1], d[~self.high].mean(0) - d[self.high].mean(0))
        self.assertAlmostEqual(outcome[0, -1], 17 / 3)
        draws = bootstrap(self.cube, self.high, 321, np.random.default_rng(42), batch_size=40)
        self.assertEqual(draws.shape, (321, 2, 7))
        np.testing.assert_allclose(draws[:, 1, :4], 2 * draws[:, 0, :4] + 3)
        np.testing.assert_allclose(draws[:, 1, 4:], 2 * draws[:, 0, 4:])
        np.testing.assert_allclose(draws[:, :, 6], draws[:, :, 5] - draws[:, :, 4])

    def test_shared_permutations_and_plus_one(self):
        d = self.cube[:, :, 0] - self.cube[:, :, 1]
        p, counts, draws = permutation_test(d, self.high, 517, np.random.default_rng(91), batch_size=53)
        np.testing.assert_allclose(draws[:, 1], 2 * draws[:, 0])
        np.testing.assert_array_equal(counts[0], counts[1])
        np.testing.assert_allclose(p, (counts + 1) / 518)
        self.assertTrue(np.all((p > 0) & (p <= 1)))

    def test_holm_known_values_and_order(self):
        raw = np.array([.04, .001, .03])
        np.testing.assert_allclose(holm_adjust(raw), [.06, .003, .06])

    def test_zero_differences_do_not_produce_significance(self):
        identical = np.zeros((6, 2))
        p, _, _ = permutation_test(identical, self.high, 23, np.random.default_rng(11))
        np.testing.assert_array_equal(p, [1., 1.])

    def test_leave_one_out_matches_direct_deletion(self):
        # Three invented resource groups with enough observations for SD output.
        identifiers = [f"synthetic_{i}" for i in range(9)]
        high = np.array([True] * 3 + [False] * 6)
        very_low = np.array([False] * 3 + [True] * 3 + [False] * 3)
        cube = np.arange(54, dtype=float).reshape(9, 3, 2)
        cube[:, :, 1] -= np.arange(9)[:, None] ** 2 / 10
        rows = []
        for i, language in enumerate(identifiers):
            for m, model in enumerate(MODEL_IDS):
                for k, direction in enumerate(("into_english", "from_english")):
                    rows.append({"model": model, "language": language,
                                 "resource_group": "high" if high[i] else "low",
                                 "direction": direction, "score": cube[i, m, k]})
        robustness, loo, _, _ = additional_analyses(pd.DataFrame(rows), cube, high, identifiers, very_low)
        np.testing.assert_allclose(robustness["primary_delta_gap"], robustness["script_weighted_delta_gap"])
        for i, language in enumerate(identifiers):
            direct = group_outcomes(np.delete(cube, i, axis=0), np.delete(high, i))[:, -1]
            actual = loo[loo["excluded_language"].eq(language)].set_index("model").loc[list(MODEL_IDS), "delta_gap"]
            np.testing.assert_allclose(actual, direct, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
