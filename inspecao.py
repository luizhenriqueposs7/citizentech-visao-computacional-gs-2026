import base64
import io
import json
import math
import tempfile
import time
from pathlib import Path

import cv2
from flask import Blueprint, Response, render_template, request, stream_with_context
from PIL import Image, ImageOps
from werkzeug.exceptions import RequestEntityTooLarge

from visao import MODELO_VISUAL, carregar_modelo, classificar


inspecao = Blueprint("inspecao", __name__)
MAX_IMAGENS = 24
MAX_QUADROS = 180
MAX_PIXELS = 4096 * 4096


def quadros_imagens(imagens):
    for i, arquivo in enumerate(imagens):
        try:
            with Image.open(arquivo.stream) as imagem:
                if imagem.format not in {"JPEG", "PNG"} or imagem.width * imagem.height > MAX_PIXELS:
                    raise ValueError("Use imagens JPEG ou PNG de até 16,8 megapixels.")
                yield arquivo.filename, ImageOps.exif_transpose(imagem).convert("RGB"), i < 6
        except (OSError, Image.DecompressionBombError) as erro:
            raise ValueError("Uma das imagens está ilegível ou excede o limite de tamanho.") from erro


def quadros_video(video):
    sufixo = Path(video.filename).suffix.lower()
    if sufixo not in {".mp4", ".mov", ".avi"}:
        raise ValueError("Use um vídeo MP4, MOV ou AVI.")
    with tempfile.TemporaryDirectory() as pasta:
        caminho = Path(pasta) / ("entrada" + sufixo)
        video.save(caminho)
        captura = cv2.VideoCapture(str(caminho))
        try:
            if not captura.isOpened():
                raise ValueError("O vídeo está ilegível ou o codec é incompatível.")
            estimativa = captura.get(cv2.CAP_PROP_FRAME_COUNT)
            passo = max(1, math.ceil(min(estimativa, MAX_QUADROS) / 6)) if math.isfinite(estimativa) else 1
            for numero in range(1, MAX_QUADROS + 2):
                ok, bgr = captura.read()
                if not ok:
                    break
                if numero > MAX_QUADROS:
                    raise ValueError(f"Envie um vídeo de até {MAX_QUADROS} quadros.")
                if bgr.shape[0] * bgr.shape[1] > MAX_PIXELS:
                    raise ValueError("O vídeo excede o limite de resolução por quadro.")
                rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                yield f"Quadro {numero:03d}", rgb, (numero - 1) % passo == 0 and (numero - 1) // passo < 6
        finally:
            captura.release()


def acompanhar(quadros, caminho_modelo=MODELO_VISUAL, ao_vivo=False):
    inicio = time.perf_counter()
    modelo = None
    resultados = []
    resolucoes = set()
    for nome, imagem, previa in quadros:
        if modelo is None:
            if not caminho_modelo.is_file():
                raise FileNotFoundError("Gere o modelo com python train.py antes de analisar.")
            inicio_carga = time.perf_counter()
            modelo, dados, preprocessar = carregar_modelo(caminho_modelo)
            inicio += time.perf_counter() - inicio_carga
        resultado = classificar(imagem, modelo, dados, preprocessar)
        resolucoes.add(f"{imagem.width} × {imagem.height}")
        resultado["nome"] = nome
        if previa or ao_vivo:
            imagem.thumbnail((340, 340))
            buffer = io.BytesIO()
            imagem.save(buffer, format="JPEG", quality=88)
            resultado["imagem"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        if ao_vivo:
            yield {
                "tipo": "quadro", "total": len(resultados) + 1,
                "resultado": resultado,
            }
            if not previa:
                resultado = {k: v for k, v in resultado.items() if k != "imagem"}
        resultados.append(resultado)
    segundos = time.perf_counter() - inicio
    if not resultados:
        raise ValueError("Nenhum quadro legível foi encontrado.")
    yield {"tipo": "concluido", "lote": {
        "resultados": resultados,
        "total": len(resultados),
        "sinalizados": sum(r["alerta"] for r in resultados),
        "fps": len(resultados) / segundos,
        "latencia_ms": 1000 * segundos / len(resultados),
        "resolucoes": ", ".join(sorted(resolucoes)),
        "entrada_modelo": dados["image_size"],
        "limiar": dados["threshold"],
    }}


def processar(quadros, caminho_modelo=MODELO_VISUAL):
    for evento in acompanhar(quadros, caminho_modelo):
        lote = evento["lote"]
    return lote


def transmitir(quadros):
    try:
        for evento in acompanhar(quadros, ao_vivo=True):
            if evento["tipo"] == "concluido":
                evento = {
                    "tipo": "concluido",
                    "html": render_template("inspecao.html", lote=evento["lote"]),
                }
            yield json.dumps(evento, ensure_ascii=False) + "\n"
    except (ValueError, FileNotFoundError) as erro:
        yield json.dumps({"tipo": "erro", "mensagem": str(erro)}, ensure_ascii=False) + "\n"
    finally:
        quadros.close()


@inspecao.route("/inspecao-visual", methods=["GET", "POST"])
def visual():
    if request.method == "GET":
        return render_template("inspecao.html")
    try:
        imagens = [f for f in request.files.getlist("imagens") if f.filename]
        video = request.files.get("video")
        if video and imagens:
            raise ValueError("Envie imagens ou um vídeo em cada análise.")
        if not video and not 1 <= len(imagens) <= MAX_IMAGENS:
            raise ValueError(f"Envie de 1 a {MAX_IMAGENS} imagens ou um vídeo curto.")
        quadros = quadros_video(video) if video else quadros_imagens(imagens)
        if request.headers.get("Accept") == "application/x-ndjson":
            return Response(stream_with_context(transmitir(quadros)), mimetype="application/x-ndjson")
        try:
            lote = processar(quadros)
        finally:
            quadros.close()
    except RequestEntityTooLarge:
        return render_template("inspecao.html", erro="O envio ultrapassou 24 MB."), 413
    except ValueError as erro:
        return render_template("inspecao.html", erro=str(erro)), 400
    except FileNotFoundError as erro:
        return render_template("inspecao.html", erro=str(erro)), 503
    return render_template("inspecao.html", lote=lote)
