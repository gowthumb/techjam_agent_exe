import unittest

from agent.one_k_policy import paired_seed_result, plateau_reached, validated_target_reached


class OneKPolicyTest(unittest.TestCase):
    def test_paired_result_accepts_consistent_practical_improvement(self):
        paired = paired_seed_result((0.6460, 0.6462, 0.6459), (0.6440, 0.6441, 0.6439))
        self.assertTrue(paired.accepted)
        self.assertGreater(paired.lower_bound, 0.0)
        self.assertAlmostEqual(paired.mean_delta, 0.002033333333333333)

    def test_paired_result_rejects_noisy_or_impractical_improvement(self):
        noisy = paired_seed_result((0.6470, 0.6430, 0.6460), (0.6440, 0.6440, 0.6440))
        impractical = paired_seed_result((0.6445, 0.6446, 0.6445), (0.6440, 0.6440, 0.6440))
        self.assertFalse(noisy.accepted)
        self.assertFalse(impractical.accepted)

    def test_validated_target_requires_paired_win_and_two_sigma_threshold(self):
        paired = paired_seed_result((0.6460, 0.6462, 0.6459), (0.6440, 0.6441, 0.6439))
        self.assertTrue(validated_target_reached((0.6460, 0.6462, 0.6459), 0.6440, 0.0008, paired))
        self.assertFalse(validated_target_reached((0.6450, 0.6451, 0.6450), 0.6440, 0.0008, paired))

    def test_plateau_requires_six_failures_and_no_remaining_direction(self):
        self.assertFalse(plateau_reached(5, False))
        self.assertFalse(plateau_reached(6, True))
        self.assertTrue(plateau_reached(6, False))


if __name__ == "__main__":
    unittest.main()