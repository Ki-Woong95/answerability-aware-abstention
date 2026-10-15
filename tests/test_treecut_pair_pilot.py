import random, unittest
from scripts.audits.treecut_pairs import build_pair, reduce_system


class TreeCutPairTests(unittest.TestCase):
    def test_pairing_and_target_indeterminacy_across_structures(self):
        for n, d in [(2, 2), (4, 2), (4, 4)]:
            for names in [False, True]:
                for cut in range(1, d):
                    for seed in [17, 29, 43]:
                        with self.subTest(n=n, d=d, names=names, cut=cut, seed=seed):
                            state = random.getstate()
                            p = build_pair(seed, n, d, names, cut)
                            self.assertEqual(random.getstate(), state)
                            a, q = p["answerable"]["problem"].rsplit("Question:", 1)
                            u, q2 = p["unanswerable"]["problem"].rsplit("Question:", 1)
                            self.assertEqual(q, q2)
                            a = a.strip().split(". ")
                            u = u.strip().split(". ")
                            self.assertEqual(len(a), n)
                            self.assertEqual(len(u), n - 1)
                            self.assertEqual(p["proof_audit"]["A_rank"], n)
                            self.assertEqual(p["proof_audit"]["U_rank"], n - 1)
                            self.assertEqual(p, build_pair(seed, n, d, names, cut))
                            self.assertEqual(
                                [
                                    x.rstrip(".")
                                    for x in a
                                    if x.rstrip(".")
                                    != p["deleted_condition"].rstrip(".")
                                ],
                                [x.rstrip(".") for x in u],
                            )

    def test_exact_solver_rejects_inconsistent_system(self):
        with self.assertRaises(ValueError):
            reduce_system([[1, 1], [2, 2]], [1, 3])


if __name__ == "__main__":
    unittest.main()
