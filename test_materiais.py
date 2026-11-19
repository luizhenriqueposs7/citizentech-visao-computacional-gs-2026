import tempfile
import unittest
import zipfile
from pathlib import Path

from baixar_materiais import destino_seguro, extrair


class TestMateriais(unittest.TestCase):
    def test_rejeita_caminhos_fora_dos_materiais(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            for nome in ("../fora.txt", "data/../../fora.txt", "data\\..\\fora.txt", "/data/foto.jpg", "app.py"):
                with self.subTest(nome=nome), self.assertRaises(ValueError):
                    destino_seguro(nome, raiz)

    def test_extrai_sem_sobrescrever_arquivo_diferente(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            pacote = raiz / "materiais.zip"
            with zipfile.ZipFile(pacote, "w") as z:
                z.writestr("data/manifesto.csv", "original")
            extrair(pacote, raiz)
            extrair(pacote, raiz)
            arquivo = raiz / "data/manifesto.csv"
            self.assertEqual(arquivo.read_text(), "original")
            arquivo.write_text("dados locais")
            with self.assertRaises(FileExistsError):
                extrair(pacote, raiz)
            self.assertEqual(arquivo.read_text(), "dados locais")


if __name__ == "__main__":
    unittest.main()
