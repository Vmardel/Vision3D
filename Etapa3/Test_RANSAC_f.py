import unittest
import numpy as np
from RANSAC_f import normalizar_puntos, estimar_F_8p, calcular_error_sampson

class TestRansac(unittest.TestCase):

    def test_normalizar_puntos(self):
        pts = np.array([[1, 1], [2, 2], [3, 3]])
        pts_norm, T = normalizar_puntos(pts)
        print("Puntos normalizados:\n", pts_norm)
        print("Matriz T de normalización:\n", T)
        self.assertEqual(pts_norm.shape, (3, 2))
        self.assertEqual(T.shape, (3, 3))

    def test_estimar_F_8p(self):
        pts1 = np.random.rand(8, 2) * 100
        pts2 = pts1 + np.random.normal(0, 1, pts1.shape)
        F = estimar_F_8p(pts1, pts2)
        print("Matriz fundamental estimada F:\n", F)
        print("Rango de F:", np.linalg.matrix_rank(F))
        self.assertEqual(F.shape, (3, 3))
        self.assertAlmostEqual(np.linalg.matrix_rank(F), 2)

    def test_error_sampson_basico(self):
        pts1 = np.random.rand(10, 2) * 100
        pts2 = pts1 + np.random.normal(0, 0.5, pts1.shape)
        F = estimar_F_8p(pts1, pts2)
        errores = calcular_error_sampson(F, pts1, pts2)
        print("Errores de Sampson:\n", errores)
        self.assertEqual(errores.shape, (10,))
        self.assertTrue(np.all(errores >= 0))

if __name__ == "__main__":
    unittest.main()

