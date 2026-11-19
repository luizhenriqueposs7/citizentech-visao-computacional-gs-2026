from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms


MODELO_VISUAL = Path(__file__).resolve().parent / "artifacts" / "crack_resnet18.pth"
IMAGE_SIZE = 160
THREADS_CPU = 4


def criar_modelo(weights=None):
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def preprocessamento(tamanho, treino=False):
    etapas = [transforms.Resize((tamanho, tamanho))]
    if treino:
        etapas.append(transforms.RandomHorizontalFlip())
    etapas.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transforms.Compose(etapas)


@lru_cache(maxsize=2)
def carregar_modelo(caminho=MODELO_VISUAL):
    torch.set_num_threads(THREADS_CPU)
    dispositivo = torch.device("cpu")
    dados = torch.load(caminho, map_location=dispositivo, weights_only=True)
    modelo = criar_modelo()
    modelo.load_state_dict(dados["state_dict"])
    return modelo.eval(), dados, preprocessamento(dados["image_size"])


def classificar(imagem, modelo, dados, preprocessar):
    tensor = preprocessar(imagem.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        escores = torch.softmax(modelo(tensor), dim=1)[0]
    escore = float(escores[dados["positive_idx"]])
    alerta = escore >= dados["threshold"]
    return {
        "classe": "Possível fissura" if alerta else "Sem fissura sinalizada",
        "confianca": escore if alerta else 1.0 - escore,
        "escore": escore,
        "alerta": alerta,
    }
