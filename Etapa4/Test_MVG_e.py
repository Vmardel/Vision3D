import unittest
import numpy as np
from MVG_e import calcular_matriz_esencial

class TestMVG(unittest.TestCase):

    def test_matriz_esencial_con_identidad(self):
        F = np.eye(3) * 0.001
        K = np.eye(3)
        E = calcular_matriz_esencial(F, K)
        print("Matriz esencial E (K = I):\n", E)
        print("Norma de E:", np.linalg.norm(E))
        self.assertEqual(E.shape, (3, 3))
        self.assertAlmostEqual(np.linalg.norm(E), 1.0, places=5)

    def test_simetria_basica(self):
        F = np.random.rand(3, 3) * 0.01
        K = np.diag([800, 800, 1])
        E = calcular_matriz_esencial(F, K)
        print("Matriz esencial E con K reales:\n", E)
        self.assertFalse(np.allclose(E, E.T))

if __name__ == "__main__":
    unittest.main()
