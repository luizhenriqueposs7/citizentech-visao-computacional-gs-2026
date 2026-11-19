import argparse
import csv
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights

from dados import CLASSES, PARTES, ler_manifesto
from visao import IMAGE_SIZE, THREADS_CPU, criar_modelo, preprocessamento


RAIZ = Path(__file__).resolve().parent
SEED = 42
BATCH_SIZE = 32
POSITIVE_IDX = CLASSES.index("crack")


class Recortes(Dataset):
    def __init__(self, pasta, registros, transformacao):
        self.pasta = pasta
        self.registros = registros
        self.transformacao = transformacao

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, indice):
        linha = self.registros[indice]
        with Image.open(self.pasta / linha["arquivo"]) as imagem:
            tensor = self.transformacao(imagem.convert("RGB"))
        return tensor, CLASSES.index(linha["classe"])


def get_probabilities(model, loader, device, positive_idx):
    """Retorna os rótulos reais e a probabilidade prevista de fissura."""
    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            crack_probabilities = probabilities[:, positive_idx]

            all_labels.extend(labels.numpy().tolist())
            all_probabilities.extend(
                crack_probabilities.cpu().numpy().tolist()
            )

    return np.asarray(all_labels), np.asarray(all_probabilities)


def choose_threshold(y_true, crack_probabilities, positive_idx):
    """
    Escolhe o limiar que maximiza F1 na validação.
    Este limiar é escolhido na validação, nunca no conjunto de teste.
    """
    y_true_binary = (y_true == positive_idx).astype(int)
    thresholds = np.arange(0.10, 0.91, 0.01)

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (crack_probabilities >= threshold).astype(int)
        score = f1_score(
            y_true_binary,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def measure_model_latency(model, image_tensor, device, repetitions=100):
    """Mede a inferência do modelo em uma imagem por vez."""
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        # Aquecimento para não medir a primeira execução isoladamente.
        for _ in range(10):
            model(image_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(repetitions):
            model(image_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

    total_ms = (time.perf_counter() - start) * 1000
    return total_ms / repetitions


def avaliar(rotulos, escores, limiar):
    real = (rotulos == POSITIVE_IDX).astype(int)
    previsto = (escores >= limiar).astype(int)
    return {
        "precision": float(precision_score(real, previsto, zero_division=0)),
        "recall": float(recall_score(real, previsto, zero_division=0)),
        "f1": float(f1_score(real, previsto, zero_division=0)),
        "accuracy": float(accuracy_score(real, previsto)),
        "matriz_confusao": confusion_matrix(real, previsto, labels=[0, 1]).tolist(),
    }


def treinar(args):
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS_CPU)
    device = torch.device("cpu")
    registros = ler_manifesto(args.dados)
    por_parte = {parte: [r for r in registros if r["conjunto"] == parte] for parte in PARTES}
    datasets = {
        parte: Recortes(args.dados, linhas, preprocessamento(args.resolucao, parte == "treino"))
        for parte, linhas in por_parte.items()
    }
    loaders = {
        parte: DataLoader(dados, batch_size=BATCH_SIZE, shuffle=parte == "treino")
        for parte, dados in datasets.items()
    }
    if args.pesos_imagenet:
        base = ResNet18_Weights.DEFAULT
        model = criar_modelo()
        estado = torch.load(args.pesos_imagenet, map_location=device, weights_only=True)
        estado = {k: v for k, v in estado.items() if not k.startswith("fc.")}
        resultado = model.load_state_dict(estado, strict=False)
        if set(resultado.missing_keys) != {"fc.weight", "fc.bias"} or resultado.unexpected_keys:
            raise ValueError(f"Pesos ImageNet incompatíveis: {resultado}")
    else:
        base = ResNet18_Weights.DEFAULT
        model = criar_modelo(base)

    model.requires_grad_(False)
    model.layer4.requires_grad_(True)
    model.fc.requires_grad_(True)
    # Estatísticas BatchNorm do ImageNet ficam fixas durante o ajuste parcial.
    model.eval()
    optimizer = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 5e-4},
    ])
    criterio = nn.CrossEntropyLoss()
    melhor = -1.0
    historico = []
    inicio = time.perf_counter()
    for epoca in range(1, args.epocas + 1):
        perdas = []
        for imagens, rotulos in loaders["treino"]:
            optimizer.zero_grad()
            loss = criterio(model(imagens), rotulos)
            loss.backward()
            optimizer.step()
            perdas.append(loss.item())

        rotulos_val, escores_val = get_probabilities(model, loaders["validacao"], device, POSITIVE_IDX)
        limiar, f1 = choose_threshold(rotulos_val, escores_val, POSITIVE_IDX)
        registro = {"epoca": epoca, "perda": float(np.mean(perdas)), "f1": f1, "limiar": limiar}
        historico.append(registro)
        print(registro, flush=True)
        if f1 > melhor:
            melhor = f1
            selecionada = epoca
            estado_final = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tempo_treino = time.perf_counter() - inicio

    model.load_state_dict(estado_final)
    rotulos_val, escores_val = get_probabilities(model, loaders["validacao"], device, POSITIVE_IDX)
    limiar, _ = choose_threshold(rotulos_val, escores_val, POSITIVE_IDX)
    rotulos, escores = get_probabilities(model, loaders["teste"], device, POSITIVE_IDX)
    latencia = measure_model_latency(model, datasets["teste"][0][0].unsqueeze(0), device)
    metricas = {
        "fonte": "10.17632/jwsn7tfbrp.1",
        "seed": SEED,
        "classes_matriz": ["no_crack", "crack"],
        "limiar": limiar,
        "epoca_selecionada": selecionada,
        "pesos_iniciais": base.name,
        "historico_validacao": historico,
        "validacao_limiar_selecionado": avaliar(rotulos_val, escores_val, limiar),
        "validacao_limiar_050": avaliar(rotulos_val, escores_val, 0.50),
        "teste": avaliar(rotulos, escores, limiar),
        "latencia_modelo_ms": latencia,
        "fps_modelo_estimado": 1000 / latencia,
        "tempo_treino_s": tempo_treino,
        "ambiente": {
            "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
            "sistema": platform.platform(), "python": platform.python_version(),
            "torch": str(torch.__version__), "threads": THREADS_CPU,
            "entrada": f"{args.resolucao}x{args.resolucao}",
        },
        "divisao": {
            parte: {
                "fotos": len({r["origem"] for r in linhas}),
                "recortes": len(linhas),
                "classes": dict(Counter(r["classe"] for r in linhas)),
            }
            for parte, linhas in por_parte.items()
        },
    }
    args.saida.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": estado_final,
        "classes": list(CLASSES),
        "positive_idx": POSITIVE_IDX,
        "negative_idx": 1 - POSITIVE_IDX,
        "threshold": limiar,
        "image_size": args.resolucao,
    }, args.saida / "crack_resnet18.pth")
    (args.saida / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8"
    )
    with (args.saida / "previsoes_teste.csv").open("w", encoding="utf-8", newline="") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(["arquivo", "origem", "classe_real", "classe_prevista", "escore_fissura"])
        for linha, real, escore in zip(por_parte["teste"], rotulos, escores, strict=True):
            previsto = POSITIVE_IDX if escore >= limiar else 1 - POSITIVE_IDX
            writer.writerow([linha["arquivo"], linha["origem"], CLASSES[real], CLASSES[previsto], f"{escore:.8f}"])
    print(json.dumps(metricas, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Treina e avalia ResNet18 com divisão auditada.")
    cli.add_argument("--dados", type=Path, default=RAIZ / "data")
    cli.add_argument("--saida", type=Path, default=RAIZ / "artifacts")
    cli.add_argument("--pesos-imagenet", type=Path)
    cli.add_argument("--resolucao", type=int, choices=(160, 224), default=IMAGE_SIZE)
    cli.add_argument("--epocas", type=int, choices=range(1, 21), default=4, metavar="1–20")
    treinar(cli.parse_args())
