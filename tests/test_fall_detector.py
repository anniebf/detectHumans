"""
Testes da lógica de queda com poses SINTÉTICAS (não precisa de câmera nem MediaPipe).

Ideia: geramos uma "pessoa palito" com ombros e quadris em posições controladas
e simulamos cenários quadro a quadro (30 FPS) com timestamps artificiais.

Rodar:  python -m pytest tests -v
"""

import math

import numpy as np
import pytest

from sentinela_pose.config import ConfigPose, ConfigQueda
from sentinela_pose.fall_detector import DetectorQueda, Estado, extrair_medidas
from sentinela_pose.pontos import OMBRO_D, OMBRO_E, QUADRIL_D, QUADRIL_E, TOTAL_PONTOS, Pose

FPS = 30
TRONCO = 80.0  # comprimento do tronco em pixels


def pose_sintetica(quadril_x: float, quadril_y: float, angulo_graus: float) -> Pose:
    """Cria uma pose com o tronco inclinado `angulo_graus` em relação à vertical."""
    pontos = np.zeros((TOTAL_PONTOS, 3), dtype=np.float32)
    pontos[:, 2] = 0.0  # pontos não usados ficam invisíveis
    a = math.radians(angulo_graus)
    ombro_x = quadril_x + TRONCO * math.sin(a)
    ombro_y = quadril_y - TRONCO * math.cos(a)
    for idx, (x, y) in {
        QUADRIL_E: (quadril_x - 10, quadril_y), QUADRIL_D: (quadril_x + 10, quadril_y),
        OMBRO_E: (ombro_x - 15, ombro_y), OMBRO_D: (ombro_x + 15, ombro_y),
    }.items():
        pontos[idx] = (x, y, 1.0)
    return Pose(pontos=pontos)


def rodar(detector, quadros, t0=0.0):
    """Executa uma sequência de poses (ou None = pessoa não vista). Devolve alertas e último estado."""
    alertas, pessoas = [], []
    for i, pose in enumerate(quadros):
        pessoas, novos = detector.processar([pose] if pose else [], t0 + i / FPS)
        alertas += novos
    return alertas, pessoas


def repetir(pose, segundos):
    return [pose] * int(segundos * FPS)


def cair(segundos=0.5, y_ini=180.0, y_fim=300.0, x=320.0):
    """Interpola de em pé (0°) para deitado (90°) com o quadril descendo rápido."""
    n = int(segundos * FPS)
    return [pose_sintetica(x, y_ini + (y_fim - y_ini) * k / n, 90 * k / n) for k in range(1, n + 1)]


@pytest.fixture
def detector():
    return DetectorQueda(640, 360, ConfigQueda(), ConfigPose())


# --------------------------- geometria ---------------------------

@pytest.mark.parametrize("angulo", [0, 30, 60, 90])
def test_angulo_tronco(angulo):
    m = extrair_medidas(pose_sintetica(320, 200, angulo))
    assert m.angulo_tronco == pytest.approx(angulo, abs=1.0)


def test_pose_sem_quadril_e_descartada():
    pose = pose_sintetica(320, 200, 0)
    pose.pontos[[QUADRIL_E, QUADRIL_D], 2] = 0.0
    assert extrair_medidas(pose) is None


# --------------------------- cenários ---------------------------

def test_em_pe_nao_alerta(detector):
    alertas, pessoas = rodar(detector, repetir(pose_sintetica(320, 180, 5), 10))
    assert alertas == []
    assert pessoas[0].estado == Estado.NORMAL


def test_queda_brusca_gera_alerta_queda(detector):
    deitado = pose_sintetica(320, 300, 90)
    seq = repetir(pose_sintetica(320, 180, 0), 1) + cair(0.5) + repetir(deitado, 3)
    alertas, _ = rodar(detector, seq)
    assert [a.tipo for a in alertas] == ["queda"]


def test_deitar_devagar_alerta_no_chao_so_apos_tempo(detector):
    # Desce em 4 s (lento, como deitar no chão de propósito ou queda lenta).
    deitado = pose_sintetica(320, 300, 90)
    seq = repetir(pose_sintetica(320, 180, 0), 1) + cair(4.0) + repetir(deitado, 5)
    alertas, _ = rodar(detector, seq)
    assert alertas == []  # ainda não passou QUEDA_TEMPO_NO_CHAO (8 s)

    alertas, _ = rodar(detector, repetir(deitado, 5), t0=len(seq) / FPS)
    assert [a.tipo for a in alertas] == ["pessoa_no_chao"]


def test_levantar_rapido_cancela(detector):
    # Cai mas se levanta antes do tempo de confirmação (2 s) -> sem alerta.
    # O levantar é gradual (1 s): na vida real o quadril não "teleporta" entre quadros.
    levantar = list(reversed(cair(1.0)))
    seq = (repetir(pose_sintetica(320, 180, 0), 1) + cair(0.5)
           + repetir(pose_sintetica(320, 300, 90), 0.3) + levantar + repetir(pose_sintetica(320, 180, 0), 3))
    alertas, pessoas = rodar(detector, seq)
    assert alertas == []
    assert pessoas[0].estado == Estado.NORMAL


def test_alerta_dispara_uma_vez_por_episodio(detector):
    seq = repetir(pose_sintetica(320, 180, 0), 1) + cair(0.5) + repetir(pose_sintetica(320, 300, 90), 20)
    alertas, _ = rodar(detector, seq)
    assert len(alertas) == 1


def test_perda_da_pose_no_chao_mantem_contagem(detector):
    # Após cair, o modelo perde a pessoa (comum com corpo deitado). A contagem deve continuar.
    seq = repetir(pose_sintetica(320, 180, 0), 1) + cair(0.5) + [None] * (3 * FPS)
    alertas, _ = rodar(detector, seq)
    assert [a.tipo for a in alertas] == ["queda"]


def test_sentar_nao_alerta(detector):
    # Sentar: quadril desce, mas o tronco continua quase vertical (< 35°).
    n = FPS
    sentar = [pose_sintetica(320, 180 + 60 * k / n, 15 * k / n) for k in range(1, n + 1)]
    alertas, _ = rodar(detector, repetir(pose_sintetica(320, 180, 0), 1) + sentar
                       + repetir(pose_sintetica(320, 240, 15), 10))
    assert alertas == []


def test_duas_pessoas_ids_independentes(detector):
    # Pessoa 1 cai; pessoa 2 (cuidador) fica em pé ao lado. Só a 1 deve alertar.
    alertas = []
    em_pe_2 = pose_sintetica(500, 180, 0)
    seq_1 = repetir(pose_sintetica(150, 180, 0), 1) + cair(0.5, x=150) + repetir(pose_sintetica(150, 300, 90), 3)
    for i, p1 in enumerate(seq_1):
        _, novos = detector.processar([p1, em_pe_2], i / FPS)
        alertas += novos
    assert len(alertas) == 1
    assert alertas[0].tipo == "queda"
