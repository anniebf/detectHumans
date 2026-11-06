"""
pontos.py - Estrutura de dados da pose e índices dos pontos do corpo.

Fica separado do pose_estimator.py para que a lógica de queda (e os testes)
não dependam do MediaPipe instalado: basta montar um array (33, 3).
"""

from dataclasses import dataclass

import numpy as np

# Índices dos pontos do MediaPipe BlazePose (33 pontos no total).
NARIZ = 0
OMBRO_E, OMBRO_D = 11, 12
COTOVELO_E, COTOVELO_D = 13, 14
PULSO_E, PULSO_D = 15, 16
QUADRIL_E, QUADRIL_D = 23, 24
JOELHO_E, JOELHO_D = 25, 26
TORNOZELO_E, TORNOZELO_D = 27, 28
TOTAL_PONTOS = 33

# Ligações entre pontos, só para desenhar o "esqueleto" na tela.
CONEXOES_ESQUELETO = [
    (OMBRO_E, OMBRO_D), (QUADRIL_E, QUADRIL_D),
    (OMBRO_E, QUADRIL_E), (OMBRO_D, QUADRIL_D),
    (QUADRIL_E, JOELHO_E), (JOELHO_E, TORNOZELO_E),
    (QUADRIL_D, JOELHO_D), (JOELHO_D, TORNOZELO_D),
    (OMBRO_E, COTOVELO_E), (COTOVELO_E, PULSO_E),
    (OMBRO_D, COTOVELO_D), (COTOVELO_D, PULSO_D),
]


@dataclass
class Pose:
    """
    Pose de UMA pessoa em UM quadro.

    pontos: array (33, 3) com [x_pixel, y_pixel, visibilidade 0..1].
            Coordenadas em pixels para que ângulos não sejam distorcidos
            pela proporção 16:9 da imagem. Eixo y cresce PARA BAIXO.
    """
    pontos: np.ndarray
