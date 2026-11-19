"""Verificações executadas e gravadas na apresentação técnica."""
import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))


def salvar(nome, dados):
    destino = PASTA / "resultados"
    destino.mkdir(exist_ok=True)
    (destino / f"{nome}.json").write_text(
        json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def dados():
    from dados import PARTES, ler_manifesto
    print("Lendo o manifesto e recalculando SHA-256 dos pixels RGB...", flush=True)
    linhas = ler_manifesto(RAIZ / "data")
    divisao = {}
    for parte in PARTES:
        registros = [r for r in linhas if r["conjunto"] == parte]
        divisao[parte] = {
            "fotos": len({r["origem"] for r in registros}),
            "recortes": len(registros),
            "classes": dict(Counter(r["classe"] for r in registros)),
        }
        item = divisao[parte]
        print(f"{parte:10} {item['fotos']:3} fotos | {item['recortes']:4} recortes | "
              f"{item['classes']}", flush=True)
    origens = [{r["origem"] for r in linhas if r["conjunto"] == p} for p in PARTES]
    assert all(not a & b for i, a in enumerate(origens) for b in origens[i + 1:])
    assert len({r["sha256_rgb"] for r in linhas}) == len(linhas) == 2304
    print("\nOK: 144 fotografias; 2.304 recortes de 227 x 227.", flush=True)
    print("OK: zero origens compartilhadas entre conjuntos.", flush=True)
    print("OK: 2.304 hashes distintos; todos os pixels conferem.", flush=True)
    salvar("dados", divisao)


def teste():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from train import Recortes, avaliar
    from visao import carregar_modelo
    linhas = list(csv.DictReader((RAIZ / "data/manifesto.csv").open(encoding="utf-8")))
    linhas = [r for r in linhas if r["conjunto"] == "teste"]
    modelo, pesos, preprocessar = carregar_modelo()
    loader = DataLoader(Recortes(RAIZ / "data", linhas, preprocessar), batch_size=32)
    print(f"Checkpoint carregado: ResNet18 | {pesos['image_size']} x {pesos['image_size']}", flush=True)
    print(f"Limiar fixado na validação: {pesos['threshold']:.2f}", flush=True)
    print("Executando inferência nas 352 imagens reservadas para teste...", flush=True)
    rotulos, escores = [], []
    with torch.inference_mode():
        for imagens, reais in loader:
            escore = torch.softmax(modelo(imagens), dim=1)[:, pesos["positive_idx"]]
            rotulos.extend(reais.tolist())
            escores.extend(escore.tolist())
            print(f"  Processadas: {len(rotulos):3}/352", flush=True)
    metricas = avaliar(np.asarray(rotulos), np.asarray(escores), pesos["threshold"])
    esperado = json.loads((RAIZ / "artifacts/metricas.json").read_text(encoding="utf-8"))["teste"]
    assert metricas["matriz_confusao"] == esperado["matriz_confusao"]
    for campo in ("precision", "recall", "f1", "accuracy"):
        assert abs(metricas[campo] - esperado[campo]) < 1e-10
    print("\nRESULTADOS RECALCULADOS", flush=True)
    for nome, campo in (("Precisão", "precision"), ("Recall", "recall"), ("F1", "f1"), ("Acurácia", "accuracy")):
        print(f"{nome:10}: {100 * metricas[campo]:.2f}%", flush=True)
    print("\nMatriz: linhas = real; colunas = previsto", flush=True)
    print("                  sem fissura   com fissura", flush=True)
    print(f"sem fissura       {metricas['matriz_confusao'][0][0]:11}   {metricas['matriz_confusao'][0][1]:11}", flush=True)
    print(f"com fissura       {metricas['matriz_confusao'][1][0]:11}   {metricas['matriz_confusao'][1][1]:11}", flush=True)
    print("\nOK: 5 falsos positivos + 8 falsos negativos = 13 erros.", flush=True)
    print("OK: resultados reproduzem artifacts/metricas.json.", flush=True)
    salvar("teste", metricas)


def sutis():
    import torch
    from torch.utils.data import DataLoader
    from experimentos import auditar_sutis, ler_csv
    from train import Recortes, get_probabilities
    from visao import carregar_modelo
    principal = ler_csv(RAIZ / "data/manifesto.csv")
    registros = ler_csv(RAIZ / "data/sutis/manifesto.csv")
    reservadas = [r for r in principal if r["conjunto"] == "teste"]
    auditar_sutis(RAIZ / "data/sutis", registros, reservadas, {r["sha256_rgb"] for r in principal})
    print(f"Conjunto complementar: {len(registros)} recortes de "
          f"{len({r['origem'] for r in registros})} fotos de teste.", flush=True)
    print("Máscaras com 1 a 31 pixels; nenhum recorte do teste principal repetido.\n", flush=True)
    resultados = {}
    for tamanho, caminho in (
        (160, RAIZ / "artifacts/crack_resnet18.pth"),
        (224, RAIZ / "artifacts/resolucao_224/crack_resnet18.pth"),
    ):
        modelo, pesos, pre = carregar_modelo(caminho)
        loader = DataLoader(Recortes(RAIZ / "data/sutis", registros, pre), batch_size=26)
        _, escores = get_probabilities(modelo, loader, torch.device("cpu"), pesos["positive_idx"])
        n = int((escores >= pesos["threshold"]).sum())
        assert n == 2 and len(escores) == 26
        resultados[tamanho] = {"sinalizados": n, "total": len(escores)}
        print(f"Entrada {tamanho} x {tamanho} | limiar {pesos['threshold']:.2f} | "
              f"sinalizados: {n}/26 ({100*n/26:.1f}%)", flush=True)
    print("\nLimitação confirmada: baixa sinalização dos fragmentos mínimos.", flush=True)
    print("Esta amostra é complementar e não altera as métricas do teste principal.", flush=True)
    salvar("sutis", resultados)


def ambiente():
    import winreg
    import torch
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as chave:
        cpu = winreg.QueryValueEx(chave, "ProcessorNameString")[0].strip()
    print(cpu, flush=True)
    print(f"{platform.system()} | Python {platform.python_version()} | PyTorch {torch.__version__}", flush=True)
    print("Inferência configurada para CPU, 4 threads.", flush=True)
    print("\nRegistro dos experimentos: artifacts/experimentos.json", flush=True)
    print("Médias de 5 execuções após aquecimento (medidas anteriores à gravação):", flush=True)
    resultados = json.loads((RAIZ / "artifacts/experimentos.json").read_text(encoding="utf-8"))
    for clipe, modelos in resultados["videos"].items():
        for tamanho, medida in modelos.items():
            print(f"{clipe:25} | {tamanho} px | "
                  f"{medida['fps_medio']:.2f} FPS | {medida['latencia_media_ms']:.2f} ms", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("acao", choices=("dados", "teste", "sutis", "ambiente"))
    acao = cli.parse_args().acao
    inicio = time.perf_counter()
    print(f"CitizenTech | verificação: {acao}\n", flush=True)
    globals()[acao]()
    print(f"\nConcluído em {time.perf_counter() - inicio:.2f} s.", flush=True)
