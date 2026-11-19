"""Baixa os pesos e a amostra preparados, verificando SHA-256 e os destinos."""
import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

RAIZ = Path(__file__).resolve().parent


def sha256(caminho):
    h = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar(dados, destino):
    if destino.exists():
        if sha256(destino) == dados["sha256"]:
            print("Já conferido:", destino.relative_to(RAIZ), flush=True)
            return destino
        raise FileExistsError(f"Preserve ou renomeie o arquivo com conteúdo diferente: {destino}")
    url = dados.get("url")
    if not url or not url.startswith("https://github.com/luizhenriqueposs7/"):
        raise ValueError("O endereço público dos materiais está ausente ou é inesperado.")
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".download")
    print("Baixando:", destino.name, flush=True)
    pedido = urllib.request.Request(url, headers={"User-Agent": "CitizenTech"})
    with urllib.request.urlopen(pedido, timeout=120) as origem, parcial.open("wb") as saida:
        shutil.copyfileobj(origem, saida)
    if sha256(parcial) != dados["sha256"]:
        parcial.unlink()
        raise ValueError("O download não corresponde ao SHA-256 publicado.")
    os.replace(parcial, destino)
    return destino


def destino_seguro(nome, raiz):
    partes = PurePosixPath(nome)
    permitido = nome.startswith("data/") or nome == "artifacts/resolucao_224/crack_resnet18.pth"
    destino = (raiz / nome).resolve()
    if (not permitido or partes.is_absolute() or ".." in partes.parts or "\\" in nome
            or not destino.is_relative_to(raiz.resolve())):
        raise ValueError(f"Caminho inesperado no pacote: {nome}")
    return destino


def extrair(pacote, raiz):
    with zipfile.ZipFile(pacote) as z:
        arquivos = [item for item in z.infolist() if not item.is_dir()]
        assert z.testzip() is None
        for item in arquivos:
            destino = destino_seguro(item.filename, raiz)
            if destino.exists() and hashlib.sha256(z.read(item)).hexdigest() != sha256(destino):
                raise FileExistsError(f"Preserve ou renomeie o arquivo com conteúdo diferente: {destino}")
        for item in arquivos:
            destino = destino_seguro(item.filename, raiz)
            if destino.exists():
                continue
            destino.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as origem, destino.open("wb") as saida:
                shutil.copyfileobj(origem, saida)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--completo", action="store_true", help="Inclui dados completos e o modelo comparativo de 224 px.")
    args = cli.parse_args()
    dados = json.loads((RAIZ / "materiais.json").read_text(encoding="utf-8"))
    baixar(dados["principal"], RAIZ / "artifacts/crack_resnet18.pth")
    if args.completo:
        pacote = baixar(dados["completo"], RAIZ / ".materiais/CitizenTech_Materiais_Reproducao.zip")
        extrair(pacote, RAIZ)
        print("Dados de treino/validação/teste e modelo comparativo disponíveis.", flush=True)
    else:
        print("Modelo principal pronto. Execute python app.py para abrir o CitizenTech.", flush=True)


if __name__ == "__main__":
    main()
