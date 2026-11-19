import unittest

from dados import auditar_divisao


class TestDivisao(unittest.TestCase):
    def setUp(self):
        self.linhas = [
            {"arquivo": f"{parte}/{classe}.jpg", "origem": f"foto-{parte}",
             "conjunto": parte, "classe": classe, "sha256_rgb": f"{parte}-{classe}"}
            for parte in ("treino", "validacao", "teste")
            for classe in ("crack", "no_crack")
        ]

    def test_origens_e_pixels_distintos(self):
        auditar_divisao(self.linhas)

    def test_recortes_diferentes_da_mesma_foto(self):
        self.linhas[-1]["origem"] = self.linhas[0]["origem"]
        with self.assertRaisesRegex(ValueError, "origem"):
            auditar_divisao(self.linhas)

    def test_mesmos_pixels_com_nomes_e_origens_diferentes(self):
        self.linhas[-1]["sha256_rgb"] = self.linhas[0]["sha256_rgb"]
        with self.assertRaisesRegex(ValueError, "pixels"):
            auditar_divisao(self.linhas)


if __name__ == "__main__":
    unittest.main()
