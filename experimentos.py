import argparse
import csv
import io
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from werkzeug.datastructures import FileStorage

from dados import CAMPOS, MIN_PIXELS_FISSURA, TAMANHO_RECORTE, ler_manifesto, sha256_rgb
from inspecao import processar, quadros_video
from train import RAIZ, Recortes, get_probabilities
from visao import MODELO_VISUAL, carregar_modelo


def ler_csv(caminho):
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def criar_sutis(originais, pasta, teste, hashes):
    por_foto = {r["origem"]: r for r in teste if r["classe"] == "crack"}
    lado = TAMANHO_RECORTE
    linhas = []
    (pasta / "recortes").mkdir(parents=True)
    for origem, referencia in sorted(por_foto.items()):
        with Image.open(originais / "BW" / f"{origem}.jpg") as bw:
            mascara = np.asarray(ImageOps.exif_transpose(bw).convert("L")) >= 128
        with Image.open(originais / "rgb" / f"{origem}.jpg") as rgb:
            foto = ImageOps.exif_transpose(rgb).convert("RGB")
        x, y = int(referencia["x"]), int(referencia["y"])
        if mascara.shape != (foto.height, foto.width) or (
            int(mascara[y:y + lado, x:x + lado].sum()) != int(referencia["pixels_mascara"])
        ):
            raise ValueError(f"Máscara incompatível com o manifesto principal: {origem}")
        for y in range(0, foto.height - lado + 1, lado):
            for x in range(0, foto.width - lado + 1, lado):
                n = int(mascara[y:y + lado, x:x + lado].sum())
                if not 0 < n < MIN_PIXELS_FISSURA:
                    continue
                imagem = io.BytesIO()
                foto.crop((x, y, x + lado, y + lado)).save(imagem, format="JPEG", quality=94)
                imagem.seek(0)
                with Image.open(imagem) as jpeg:
                    digest = sha256_rgb(jpeg.convert("RGB"))
                if digest in hashes:
                    continue
                hashes.add(digest)
                nome = f"recortes/{origem}_{x}_{y}.jpg"
                (pasta / nome).write_bytes(imagem.getvalue())
                linhas.append(dict(zip(CAMPOS, (nome, "crack", origem, "teste", x, y, n, digest))))
    with (pasta / "manifesto.csv").open("w", encoding="utf-8", newline="") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=CAMPOS)
        writer.writeheader()
        writer.writerows(linhas)


def auditar_sutis(pasta, linhas, teste, hashes):
    origens = {r["origem"] for r in teste}
    if not linhas:
        raise ValueError("A amostra de recortes sutis está vazia.")
    for linha in linhas:
        caminho = (pasta / linha["arquivo"]).resolve()
        if (not caminho.is_relative_to(pasta.resolve()) or linha["origem"] not in origens
                or linha["classe"] != "crack" or linha["conjunto"] != "teste"
                or not 0 < int(linha["pixels_mascara"]) < MIN_PIXELS_FISSURA):
            raise ValueError(f"Recorte sutil fora do protocolo: {linha}")
        with Image.open(caminho) as imagem:
            digest = sha256_rgb(imagem.convert("RGB"))
        if digest != linha["sha256_rgb"] or digest in hashes:
            raise ValueError(f"Recorte sutil divergente ou duplicado: {caminho}")
        hashes.add(digest)


def criar_segundo_clipe(caminho, teste, exclusoes):
    rng = random.Random(42)
    opcoes = teste.copy()
    rng.shuffle(opcoes)
    origens = {r["origem"] for r in teste if r["arquivo"] in exclusoes}
    escolhidos = []
    for classe in ("crack", "no_crack") * 3:
        linha = next(r for r in opcoes if r["classe"] == classe and r["origem"] not in origens)
        origens.add(linha["origem"])
        escolhidos.append(linha["arquivo"])
    if caminho.exists():
        return escolhidos
    video = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), 12, (640, 480))
    if not video.isOpened():
        raise RuntimeError("O codificador MP4 não abriu.")
    try:
        for nome in escolhidos:
            quadro = cv2.imread(str(RAIZ / "data" / nome))
            ampliado = cv2.resize(quadro, (690, 500), interpolation=cv2.INTER_CUBIC)
            for i in range(12):
                x = i * 4
                video.write(ampliado[10:490, x:x + 640])
    finally:
        video.release()
    return escolhidos


def medir_video(caminho, pesos):
    with caminho.open("rb") as arquivo:
        quadros = quadros_video(FileStorage(stream=arquivo, filename=caminho.name))
        try:
            resultado = processar(quadros, pesos)
        finally:
            quadros.close()
    return {campo: resultado[campo] for campo in (
        "total", "fps", "latencia_ms", "resolucoes", "sinalizados"
    )}


def main(originais):
    dados = RAIZ / "data"
    principal = ler_manifesto(dados)
    teste = [r for r in principal if r["conjunto"] == "teste"]
    hashes = {r["sha256_rgb"] for r in principal}
    pasta_sutis = dados / "sutis"
    manifesto = pasta_sutis / "manifesto.csv"
    if originais is not None:
        criar_sutis(originais, pasta_sutis, teste, hashes.copy())
    sutis = ler_csv(manifesto)
    auditar_sutis(pasta_sutis, sutis, teste, hashes)

    artifacts = RAIZ / "artifacts"
    demo = RAIZ / "demo"
    sequencia = json.loads((artifacts / "desempenho_video.json").read_text(encoding="utf-8"))["sequencia"]
    clipes = (demo / "concreto_sequencia.mp4", demo / "concreto_movimento.mp4")
    movimento = criar_segundo_clipe(clipes[1], teste, set(sequencia))
    modelos = {160: MODELO_VISUAL, 224: artifacts / "resolucao_224" / MODELO_VISUAL.name}
    complemento = {}
    for tamanho, caminho in modelos.items():
        modelo, pesos, preprocessar = carregar_modelo(caminho)
        loader = DataLoader(Recortes(pasta_sutis, sutis, preprocessar), batch_size=32)
        _, escores = get_probabilities(modelo, loader, torch.device("cpu"), pesos["positive_idx"])
        assert pesos["image_size"] == tamanho and len(escores) == len(sutis)
        complemento[tamanho] = {
            "limiar": pesos["threshold"],
            "sinalizados": int((escores >= pesos["threshold"]).sum()),
            "escores": escores.tolist(),
        }

    medidas = defaultdict(list)
    for clip in clipes:
        for _ in range(6):
            for tamanho, pesos in modelos.items():
                medidas[(clip.name, tamanho)].append(medir_video(clip, pesos))
    resultado = {
        "sutis": {
            "recortes": len(sutis), "fotos": len({r["origem"] for r in sutis}),
            "modelos": complemento,
        },
        "origens_clipes": {clipes[0].name: sequencia, clipes[1].name: movimento},
        "videos": {},
    }
    a = ler_csv(artifacts / "previsoes_teste.csv")
    b = ler_csv(artifacts / "resolucao_224" / "previsoes_teste.csv")
    transicoes = {"fissuras_recuperadas": 0, "fissuras_perdidas": 0,
                  "falsos_positivos_corrigidos": 0, "falsos_positivos_novos": 0}
    for primeiro, segundo in zip(a, b, strict=True):
        real = primeiro["classe_real"]
        assert (primeiro["arquivo"], real) == (segundo["arquivo"], segundo["classe_real"])
        mudou = primeiro["classe_prevista"] != segundo["classe_prevista"]
        if mudou:
            acertou = segundo["classe_prevista"] == real
            if real == "crack":
                chave = "fissuras_recuperadas" if acertou else "fissuras_perdidas"
            else:
                chave = "falsos_positivos_corrigidos" if acertou else "falsos_positivos_novos"
            transicoes[chave] += 1
    resultado["transicoes_de_160_para_224"] = transicoes
    for (clip, tamanho), (aquecimento, *runs) in medidas.items():
        assert {r["total"] for r in runs} == {aquecimento["total"]}
        resultado["videos"].setdefault(clip, {})[tamanho] = {
            "quadros": aquecimento["total"], "resolucao_video": aquecimento["resolucoes"],
            "fps_medio": mean(r["fps"] for r in runs),
            "latencia_media_ms": mean(r["latencia_ms"] for r in runs),
            "repeticoes": runs,
        }
    (artifacts / "experimentos.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(sutis)} recortes sutis auditados e dois clipes medidos nas duas resoluções.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Avalia recortes sutis e mede dois clipes nas duas resoluções.")
    cli.add_argument("--originais", type=Path, help="Regenera os recortes sutis a partir de rgb/ e BW/.")
    main(cli.parse_args().originais)
