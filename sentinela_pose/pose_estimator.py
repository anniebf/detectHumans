"""
pose_estimator.py - Wrapper do MediaPipe Pose Landmarker (API "Tasks").

Por que MediaPipe?
  - Roda em tempo real em CPU comum (modelo "lite" tem ~3 MB).
  - Não precisa de PyTorch/CUDA (o requirements antigo baixava torch+cu121, ~2,5 GB).
  - Devolve 33 pontos do corpo, que são muito mais informativos para queda
    do que uma caixa (bounding box).

Observação: nas versões atuais do mediapipe (>= 0.10) a API antiga
`mp.solutions.pose` não existe mais; usamos `mediapipe.tasks.python.vision`.
"""

import os
import urllib.request
from typing import List

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .config import ConfigPose
from .pontos import Pose

# URLs oficiais dos modelos publicados pelo Google.
_URL_MODELOS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def garantir_modelo(cfg: ConfigPose) -> str:
    """Baixa o arquivo .task do modelo na primeira execução e devolve o caminho local."""
    variante = cfg.variante_modelo if cfg.variante_modelo in _URL_MODELOS else "lite"
    os.makedirs(cfg.pasta_modelos, exist_ok=True)
    caminho = os.path.join(cfg.pasta_modelos, f"pose_landmarker_{variante}.task")
    if not os.path.exists(caminho):
        print(f"[Pose] Baixando modelo '{variante}' para {caminho} ...")
        urllib.request.urlretrieve(_URL_MODELOS[variante], caminho)
    return caminho


class EstimadorPose:
    """
    Recebe quadros BGR (OpenCV) e devolve uma lista de `Pose`.

    Usa o modo VIDEO do MediaPipe: ele aproveita o quadro anterior para
    rastrear a pessoa, o que é mais rápido e estável do que detectar do zero.
    Exige timestamps crescentes em milissegundos.
    """

    def __init__(self, cfg: ConfigPose | None = None):
        self.cfg = cfg or ConfigPose()
        opcoes = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=garantir_modelo(self.cfg)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=self.cfg.max_pessoas,
            min_pose_detection_confidence=self.cfg.conf_deteccao,
            min_pose_presence_confidence=self.cfg.conf_presenca,
            min_tracking_confidence=self.cfg.conf_rastreio,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opcoes)
        self._ultimo_ts_ms = -1

    def processar(self, quadro_bgr: np.ndarray, timestamp_s: float) -> List[Pose]:
        """Roda a estimativa de pose em um quadro e converte o resultado para pixels."""
        altura, largura = quadro_bgr.shape[:2]

        # MediaPipe espera RGB.
        rgb = cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB)
        imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # O modo VIDEO rejeita timestamps repetidos/decrescentes; garantimos +1 ms.
        ts_ms = max(int(timestamp_s * 1000), self._ultimo_ts_ms + 1)
        self._ultimo_ts_ms = ts_ms

        resultado = self._landmarker.detect_for_video(imagem_mp, ts_ms)

        poses: List[Pose] = []
        for marcos in resultado.pose_landmarks:
            # Converte coordenadas normalizadas (0..1) para pixels.
            pontos = np.array(
                [[m.x * largura, m.y * altura, m.visibility if m.visibility is not None else 1.0] for m in marcos],
                dtype=np.float32,
            )
            poses.append(Pose(pontos=pontos))
        return poses

    def fechar(self) -> None:
        """Libera os recursos nativos do MediaPipe."""
        self._landmarker.close()
