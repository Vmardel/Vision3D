import unittest
import numpy as np
from Factorizacion import descomposicion_rq, factorizar_P

class TestFactoriza(unittest.TestCase):

    def test_decomposicion_rq(self):
        M = np.array([[1,2,3],[0,1,4],[0,0,1]])
        K, R = descomposicion_rq(M)
        # Comprobar que K es triangular superior
        self.assertTrue(np.allclose(np.tril(K, -1), 0))
        # Comprobar que R es ortogonal: R @ R.T = I
        self.assertTrue(np.allclose(R @ R.T, np.eye(3)))

    def test_factorizar_P(self):
        P = np.hstack([np.eye(3), np.array([[1],[2],[3]])])
        K, R, t = factorizar_P(P)
        self.assertEqual(K.shape, (3,3))
        self.assertEqual(R.shape, (3,3))
        self.assertEqual(t.shape, (3,))

if __name__ == '__main__':
    unittest.main()
