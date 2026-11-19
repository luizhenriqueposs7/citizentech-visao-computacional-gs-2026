import argparse
import csv
import hashlib
import io
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


PARTES = ("treino", "validacao", "teste")
CLASSES = ("crack", "no_crack")
CAMPOS = ("arquivo", "classe", "origem", "conjunto", "x", "y", "pixels_mascara", "sha256_rgb")
TAMANHO_RECORTE = 227
MIN_PIXELS_FISSURA = 32


def auditar_divisao(linhas):
    proprietarios = {"origem": {}, "sha256_rgb": {}}
    caminhos = set()
    classes_por_parte = {parte: set() for parte in PARTES}
    for linha in linhas:
        parte = linha["conjunto"]
        caminho = linha["arquivo"]
        classe = linha["classe"]
        if parte not in PARTES or classe not in CLASSES or caminho in caminhos:
            raise ValueError(f"Linha inválida ou caminho repetido: {linha}")
        caminhos.add(caminho)
        classes_por_parte[parte].add(classe)
        for campo, grupos in proprietarios.items():
            chave = linha[campo]
            if not chave or grupos.setdefault(chave, parte) != parte:
                nome = "origem" if campo == "origem" else "pixels"
                raise ValueError(f"Vazamento de {nome} ou valor ausente em {caminho}.")
    if any(classes != set(CLASSES) for classes in classes_por_parte.values()):
        raise ValueError("Cada conjunto deve conter as duas classes.")


def ler_manifesto(pasta):
    with (pasta / "manifesto.csv").open(encoding="utf-8", newline="") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    auditar_divisao(linhas)
    for linha in linhas:
        caminho = (pasta / linha["arquivo"]).resolve()
        if not caminho.is_relative_to(pasta.resolve()):
            raise ValueError(f"Recorte fora da pasta de dados: {caminho}")
        with Image.open(caminho) as imagem:
            if sha256_rgb(imagem.convert("RGB")) != linha["sha256_rgb"]:
                raise ValueError(f"Pixels divergentes do manifesto: {caminho}")
    return linhas


def sha256_rgb(imagem):
    return hashlib.sha256(imagem.tobytes()).hexdigest()


def preparar(originais, saida, limite_fotos, por_classe, seed):
    fotos = {p.stem: p for p in (originais / "rgb").glob("*.jpg")}
    mascaras = {p.stem: p for p in (originais / "BW").glob("*.jpg")}
    if fotos.keys() != mascaras.keys() or len(fotos) < 10:
        raise ValueError("São necessários pares RGB e BW com o mesmo identificador.")
    if limite_fotos < 10 or por_classe < 1:
        raise ValueError("Use pelo menos 10 fotos e um recorte por classe.")
    if saida.exists():
        raise FileExistsError(f"Escolha uma pasta de saída nova: {saida}")

    rng = random.Random(seed)
    candidatos = sorted(fotos)
    rng.shuffle(candidatos)
    origens = []
    origens_vistas = set()
    for origem in candidatos:
        with Image.open(fotos[origem]) as foto:
            digest = sha256_rgb(ImageOps.exif_transpose(foto).convert("RGB"))
        if digest not in origens_vistas:
            origens_vistas.add(digest)
            origens.append(origem)
        if len(origens) == limite_fotos:
            break

    n_avaliacao = max(1, round(len(origens) * 0.15))
    atribuicao = {
        origem: "teste" if i < n_avaliacao else
        "validacao" if i < 2 * n_avaliacao else "treino"
        for i, origem in enumerate(origens)
    }
    lado = TAMANHO_RECORTE
    linhas = []
    pixels_vistos = set()
    destino = saida.resolve()
    destino.mkdir(parents=True)
    try:
        (destino / "recortes").mkdir()
        for origem in origens:
            with Image.open(fotos[origem]) as rgb, Image.open(mascaras[origem]) as bw:
                foto = ImageOps.exif_transpose(rgb).convert("RGB")
                bw_orientada = ImageOps.exif_transpose(bw)
                if foto.size != bw_orientada.size:
                    raise ValueError(f"Foto e máscara têm tamanhos diferentes: {origem}")
                mascara = np.asarray(bw_orientada.convert("L")) >= 128

            recortes = {classe: [] for classe in CLASSES}
            for y in range(0, foto.height - lado + 1, lado):
                for x in range(0, foto.width - lado + 1, lado):
                    n = int(mascara[y:y + lado, x:x + lado].sum())
                    if n >= MIN_PIXELS_FISSURA:
                        recortes["crack"].append((x, y, n))
                    elif n == 0:
                        recortes["no_crack"].append((x, y, n))

            for classe, opcoes in recortes.items():
                rng.shuffle(opcoes)
                incluidos = 0
                for x, y, n in opcoes:
                    recorte = foto.crop((x, y, x + lado, y + lado))
                    imagem = io.BytesIO()
                    recorte.save(imagem, format="JPEG", quality=94)
                    imagem.seek(0)
                    with Image.open(imagem) as jpeg:
                        digest = sha256_rgb(jpeg.convert("RGB"))
                    if digest in pixels_vistos:
                        continue
                    pixels_vistos.add(digest)
                    nome = f"recortes/{origem}_{x}_{y}.jpg"
                    (destino / nome).write_bytes(imagem.getvalue())
                    linhas.append(dict(zip(CAMPOS, (
                        nome, classe, origem, atribuicao[origem], x, y, n, digest
                    ))))
                    incluidos += 1
                    if incluidos == por_classe:
                        break

        auditar_divisao(linhas)
        with (destino / "manifesto.csv").open("w", encoding="utf-8", newline="") as arquivo:
            writer = csv.DictWriter(arquivo, fieldnames=CAMPOS)
            writer.writeheader()
            writer.writerows(linhas)
    except BaseException:
        shutil.rmtree(destino)
        raise

    for parte in PARTES:
        registros = [linha for linha in linhas if linha["conjunto"] == parte]
        print(parte, len({r["origem"] for r in registros}), "fotos",
              len(registros), "recortes", dict(Counter(r["classe"] for r in registros)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Gera recortes por foto original e audita a divisão.")
    cli.add_argument("originais", type=Path, help="Pasta com rgb/ e BW/ extraídas da base oficial.")
    cli.add_argument("--saida", type=Path, default=Path(__file__).resolve().parent / "data")
    cli.add_argument("--limite-fotos", type=int, default=144)
    cli.add_argument("--por-classe", type=int, default=8)
    cli.add_argument("--seed", type=int, default=42)
    args = cli.parse_args()
    preparar(args.originais, args.saida, args.limite_fotos, args.por_classe, args.seed)
