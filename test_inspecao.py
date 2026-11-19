import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

from app import app


class TestInspecao(unittest.TestCase):
    def test_quadros_sao_enviados_durante_a_execucao(self):
        imagem = io.BytesIO()
        Image.new("RGB", (40, 40), "gray").save(imagem, format="JPEG")
        metadados = {"image_size": 160, "threshold": 0.49}
        resultado = {"classe": "Possível fissura", "confianca": 0.9, "escore": 0.9, "alerta": True}
        with (
            app.test_client() as cliente,
            patch("inspecao.carregar_modelo", return_value=(None, metadados, None)),
            patch("inspecao.classificar", side_effect=lambda *args: resultado.copy()) as classificar,
        ):
            resposta = cliente.post("/inspecao-visual", data={
                "imagens": [(io.BytesIO(imagem.getvalue()), f"concreto{i}.jpg") for i in range(2)],
            }, headers={"Accept": "application/x-ndjson"}, buffered=False)
            self.assertEqual(resposta.mimetype, "application/x-ndjson")
            eventos = iter(resposta.response)
            primeiro = json.loads(next(eventos))
            self.assertEqual(primeiro["tipo"], "quadro")
            self.assertEqual(primeiro["total"], 1)
            self.assertEqual(classificar.call_count, 1)
            segundo = json.loads(next(eventos))
            self.assertEqual(segundo["total"], 2)
            self.assertEqual(classificar.call_count, 2)
            final = json.loads(next(eventos))
            self.assertEqual(final["tipo"], "concluido")
            self.assertIn("FPS observado", final["html"])
            resposta.close()

    def test_fps_apenas_no_processamento_continuo(self):
        with app.test_client() as cliente, patch("inspecao.processar") as processar:
            for total in (1, 2):
                processar.return_value = {
                    "total": total, "sinalizados": 0, "latencia_ms": 10.0, "fps": 100.0,
                    "resolucoes": "227 × 227", "entrada_modelo": 160, "limiar": 0.49,
                    "resultados": [],
                }
                resposta = cliente.post("/inspecao-visual", data={
                    "imagens": [(io.BytesIO(b"entrada"), f"concreto{i}.jpg") for i in range(total)],
                })
                self.assertEqual(resposta.status_code, 200)
                self.assertEqual(b"FPS observado" in resposta.data, total > 1)
                self.assertIn(b"10.0 ms", resposta.data)

    def test_pagina_faz_parte_do_prototipo(self):
        with app.test_client() as cliente:
            self.assertIn(b'/inspecao-visual', cliente.get("/").data)
            resposta = cliente.get("/inspecao-visual")
            self.assertEqual(resposta.status_code, 200)
            self.assertIn(b"CitizenTech", resposta.data)

    def test_imagem_invalida_tem_resposta_controlada(self):
        with app.test_client() as cliente:
            resposta = cliente.post(
                "/inspecao-visual",
                data={"imagens": (io.BytesIO(b"invalido"), "concreto.jpg")},
            )
            self.assertEqual(resposta.status_code, 400)


if __name__ == "__main__":
    unittest.main()
