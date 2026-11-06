"""
fall_detector.py - Lógica de detecção de queda a partir dos pontos da pose.

Este arquivo NÃO depende do MediaPipe: recebe `Pose` (array 33x3) e um
timestamp, e devolve o estado de cada pessoa. Por isso é testável com dados
sintéticos (ver tests/test_fall_detector.py).

Fluxo por quadro:
    poses -> RastreadorSimples (dá um ID para cada pessoa)
          -> extrair_medidas()  (ângulo do tronco, altura do quadril, ...)
          -> MaquinaEstadosQueda.atualizar()  (decide NORMAL / CAINDO / NO_CHAO / ALERTA)

Estados:
    NORMAL   : pessoa em pé, sentada ou andando.
    CAINDO   : postura deitada iniciada logo após uma descida brusca (impacto).
    NO_CHAO  : postura deitada sem impacto visto (pode ser queda lenta ou deitar).
    ALERTA   : tempo de confirmação estourou -> alerta disparado (uma vez por episódio).
"""

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import ConfigPose, ConfigQueda
from .pontos import OMBRO_D, OMBRO_E, QUADRIL_D, QUADRIL_E, Pose


class Estado(str, Enum):
    NORMAL = "normal"
    CAINDO = "caindo"
    NO_CHAO = "no_chao"
    ALERTA = "alerta"


@dataclass
class Medidas:
    """Medidas geométricas de uma pose em um quadro."""
    angulo_tronco: float          # 0° = tronco vertical, 90° = horizontal
    quadril: Tuple[float, float]  # centro dos quadris em pixels (x, y)
    tamanho_tronco: float         # distância ombro-quadril em pixels (escala da pessoa)
    proporcao_corpo: float        # altura/largura da caixa formada pelos pontos visíveis


@dataclass
class EventoAlerta:
    """Informação entregue a quem envia o alerta (WhatsApp, API, etc.)."""
    id_pessoa: int
    tipo: str            # "queda" ou "pessoa_no_chao"
    timestamp: float
    angulo_tronco: float


# ---------------------------------------------------------------------------
# 1) GEOMETRIA
# ---------------------------------------------------------------------------

def _media_pontos(pontos: np.ndarray, idx_a: int, idx_b: int, vis_min: float) -> Optional[np.ndarray]:
    """Média de dois pontos (ex.: ombro esq. e dir.). Usa só os que estiverem visíveis."""
    visiveis = [pontos[i, :2] for i in (idx_a, idx_b) if pontos[i, 2] >= vis_min]
    if not visiveis:
        return None
    return np.mean(visiveis, axis=0)


def extrair_medidas(pose: Pose, vis_min: float = 0.3) -> Optional[Medidas]:
    """
    Calcula as medidas que a máquina de estados usa.
    Retorna None se ombros ou quadris não estiverem visíveis (pose inutilizável).
    """
    p = pose.pontos
    ombro = _media_pontos(p, OMBRO_E, OMBRO_D, vis_min)
    quadril = _media_pontos(p, QUADRIL_E, QUADRIL_D, vis_min)
    if ombro is None or quadril is None:
        return None

    # Vetor quadril -> ombro. Em pé ele aponta para CIMA (y negativo, pois y cresce para baixo).
    dx, dy = ombro[0] - quadril[0], ombro[1] - quadril[1]
    tamanho_tronco = math.hypot(dx, dy)
    if tamanho_tronco < 1e-3:
        return None

    # atan2(|dx|, -dy): 0° com ombro acima do quadril, 90° na horizontal,
    # >90° se a cabeça estiver abaixo do quadril (também conta como "no chão").
    angulo = math.degrees(math.atan2(abs(dx), -dy))

    # Caixa dos pontos visíveis: medida auxiliar (só para exibição/depuração).
    visiveis = p[p[:, 2] >= vis_min, :2]
    largura = max(float(np.ptp(visiveis[:, 0])), 1.0)
    altura = max(float(np.ptp(visiveis[:, 1])), 1.0)

    return Medidas(
        angulo_tronco=angulo,
        quadril=(float(quadril[0]), float(quadril[1])),
        tamanho_tronco=tamanho_tronco,
        proporcao_corpo=altura / largura,
    )


# ---------------------------------------------------------------------------
# 2) RASTREAMENTO (dá um ID estável para cada pessoa)
# ---------------------------------------------------------------------------

class RastreadorSimples:
    """
    Associa poses entre quadros pelo vizinho mais próximo (centro do quadril).

    MediaPipe não devolve IDs; para salas com poucas pessoas (quarto de ILPI)
    a associação gulosa por distância é suficiente e custa praticamente zero.
    Se houver muitas pessoas cruzando, troque por ByteTrack/Norfair.
    """

    def __init__(self, distancia_max_px: float):
        self.distancia_max_px = distancia_max_px
        self._ultima_posicao: Dict[int, Tuple[float, float]] = {}
        self._proximo_id = 1

    def associar(self, medidas: List[Medidas]) -> List[int]:
        """Devolve um ID para cada medida, na mesma ordem da lista recebida."""
        ids: List[int] = [-1] * len(medidas)
        livres = set(self._ultima_posicao.keys())

        # Monta todos os pares (distância, índice da medida, id) e casa do mais perto para o mais longe.
        pares = []
        for i, m in enumerate(medidas):
            for id_p, pos in self._ultima_posicao.items():
                pares.append((math.dist(m.quadril, pos), i, id_p))
        for dist, i, id_p in sorted(pares):
            if dist > self.distancia_max_px:
                break
            if ids[i] == -1 and id_p in livres:
                ids[i] = id_p
                livres.discard(id_p)

        # Quem não casou com ninguém ganha um ID novo.
        for i, m in enumerate(medidas):
            if ids[i] == -1:
                ids[i] = self._proximo_id
                self._proximo_id += 1
            self._ultima_posicao[ids[i]] = m.quadril
        return ids

    def remover(self, id_pessoa: int) -> None:
        self._ultima_posicao.pop(id_pessoa, None)


# ---------------------------------------------------------------------------
# 3) MÁQUINA DE ESTADOS (uma por pessoa)
# ---------------------------------------------------------------------------

@dataclass
class MaquinaEstadosQueda:
    """Decide, ao longo do tempo, se uma pessoa caiu ou está no chão."""

    cfg: ConfigQueda
    estado: Estado = Estado.NORMAL
    # Histórico curto (timestamp, y_quadril, tamanho_tronco) para medir a velocidade de descida.
    _historico: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=60))
    _ultimo_impacto: float = -1e9      # instante da última descida brusca
    _inicio_deitado: Optional[float] = None
    _inicio_em_pe: Optional[float] = None
    _ultima_observacao: float = -1e9   # última vez que a pose foi vista
    _ultima_deitado: float = -1e9      # última vez que a pose foi vista deitada
    ultimo_angulo: float = 0.0
    ultima_velocidade: float = 0.0

    def _velocidade_descida(self, agora: float) -> float:
        """
        Velocidade de descida do quadril em 'comprimentos de tronco por segundo'.
        Positivo = descendo. Normalizar pelo tronco faz o limiar valer perto ou longe da câmera.
        """
        # Pega o ponto mais antigo dentro da janela configurada.
        antigo = None
        for ts, y, tronco in self._historico:
            if agora - ts <= self.cfg.janela_velocidade:
                antigo = (ts, y, tronco)
                break
        if antigo is None or not self._historico:
            return 0.0
        ts0, y0, tronco0 = antigo
        ts1, y1, _ = self._historico[-1]
        dt = ts1 - ts0
        if dt < 1e-3:
            return 0.0
        # Usa o tronco do início da janela (antes da queda o tronco aparece inteiro, medida mais confiável).
        return (y1 - y0) / max(tronco0, 1.0) / dt

    def atualizar(self, agora: float, medidas: Optional[Medidas]) -> Optional[str]:
        """
        Avança a máquina de estados.
        medidas=None significa "pessoa não vista neste quadro".
        Retorna "queda" ou "pessoa_no_chao" no quadro em que o alerta deve ser disparado.
        """
        cfg = self.cfg

        if medidas is not None:
            self._ultima_observacao = agora
            self._historico.append((agora, medidas.quadril[1], medidas.tamanho_tronco))
            self.ultimo_angulo = medidas.angulo_tronco
            self.ultima_velocidade = self._velocidade_descida(agora)

            # (a) Descida brusca do quadril = possível impacto.
            if self.ultima_velocidade >= cfg.velocidade_queda:
                self._ultimo_impacto = agora

            deitado = medidas.angulo_tronco >= cfg.angulo_deitado
            em_pe = medidas.angulo_tronco <= cfg.angulo_em_pe
        else:
            # Sem pose: mantém a última postura conhecida enquanto estiver dentro da tolerância.
            # Isso é importante porque o modelo às vezes perde a pessoa justamente quando ela está no chão.
            deitado = self.estado != Estado.NORMAL
            em_pe = False

        # (b) Postura deitada: inicia/continua a contagem de tempo no chão.
        if deitado:
            self._ultima_deitado = agora if medidas is not None else self._ultima_deitado
            self._inicio_em_pe = None
            if self._inicio_deitado is None:
                self._inicio_deitado = agora
                houve_impacto = (agora - self._ultimo_impacto) <= cfg.janela_impacto
                self.estado = Estado.CAINDO if houve_impacto else Estado.NO_CHAO
            elif self.estado == Estado.NO_CHAO and (agora - self._ultimo_impacto) <= cfg.janela_impacto:
                # Impacto detectado logo depois de deitar (ex.: deitou de lado e o quadril "despencou").
                self.estado = Estado.CAINDO

            tempo_deitado = agora - self._inicio_deitado
            if self.estado == Estado.CAINDO and tempo_deitado >= cfg.tempo_confirma_queda:
                self.estado = Estado.ALERTA
                return "queda"
            if self.estado == Estado.NO_CHAO and tempo_deitado >= cfg.tempo_no_chao:
                self.estado = Estado.ALERTA
                return "pessoa_no_chao"
            return None

        # (c) Em pé/ereto: só volta ao NORMAL depois de um tempo contínuo (evita piscar).
        if em_pe:
            if self._inicio_em_pe is None:
                self._inicio_em_pe = agora
            if agora - self._inicio_em_pe >= cfg.tempo_recuperacao:
                self._resetar()
            return None

        # (d) Zona intermediária (inclinado, agachado) ou falha curta: tolera sem zerar a contagem.
        if self._inicio_deitado is not None and (agora - self._ultima_deitado) > cfg.tolerancia_falha:
            # Ficou tempo demais fora da postura deitada, mas sem ficar em pé: reinicia a contagem.
            if self.estado != Estado.ALERTA:
                self._inicio_deitado = None
                self.estado = Estado.NORMAL
        return None

    def _resetar(self) -> None:
        self.estado = Estado.NORMAL
        self._inicio_deitado = None
        self._inicio_em_pe = None

    @property
    def ultima_observacao(self) -> float:
        return self._ultima_observacao


# ---------------------------------------------------------------------------
# 4) FACHADA: junta rastreador + máquinas de estado
# ---------------------------------------------------------------------------

@dataclass
class EstadoPessoa:
    """Resumo de uma pessoa para desenhar na tela / enviar à API."""
    id_pessoa: int
    estado: Estado
    angulo_tronco: float
    velocidade: float
    pose: Optional[Pose]


class DetectorQueda:
    """
    Ponto de entrada da lógica. Uso:
        detector = DetectorQueda(largura, altura)
        pessoas, alertas = detector.processar(poses, timestamp)
    """

    def __init__(self, largura: int, altura: int,
                 cfg_queda: ConfigQueda | None = None, cfg_pose: ConfigPose | None = None):
        self.cfg = cfg_queda or ConfigQueda()
        self.vis_min = (cfg_pose or ConfigPose()).visibilidade_min
        diagonal = math.hypot(largura, altura)
        self.rastreador = RastreadorSimples(self.cfg.distancia_max_rastreio * diagonal)
        self.maquinas: Dict[int, MaquinaEstadosQueda] = {}

    def processar(self, poses: List[Pose], agora: float) -> Tuple[List[EstadoPessoa], List[EventoAlerta]]:
        # 1. Extrai medidas e descarta poses sem ombro/quadril visíveis.
        validas = [(p, m) for p in poses if (m := extrair_medidas(p, self.vis_min)) is not None]
        ids = self.rastreador.associar([m for _, m in validas])

        pessoas: List[EstadoPessoa] = []
        alertas: List[EventoAlerta] = []
        vistos = set()

        # 2. Atualiza a máquina de cada pessoa vista neste quadro.
        for (pose, medidas), id_p in zip(validas, ids):
            vistos.add(id_p)
            maquina = self.maquinas.setdefault(id_p, MaquinaEstadosQueda(self.cfg))
            tipo = maquina.atualizar(agora, medidas)
            if tipo:
                alertas.append(EventoAlerta(id_p, tipo, agora, medidas.angulo_tronco))
            pessoas.append(EstadoPessoa(id_p, maquina.estado, maquina.ultimo_angulo, maquina.ultima_velocidade, pose))

        # 3. Pessoas não vistas: avança o tempo (pode disparar alerta se estavam no chão)
        #    e descarta quem sumiu há muito tempo.
        for id_p in list(self.maquinas.keys()):
            if id_p in vistos:
                continue
            maquina = self.maquinas[id_p]
            if agora - maquina.ultima_observacao > self.cfg.tempo_perda_max:
                del self.maquinas[id_p]
                self.rastreador.remover(id_p)
                continue
            tipo = maquina.atualizar(agora, None)
            if tipo:
                alertas.append(EventoAlerta(id_p, tipo, agora, maquina.ultimo_angulo))
            if maquina.estado != Estado.NORMAL:
                pessoas.append(EstadoPessoa(id_p, maquina.estado, maquina.ultimo_angulo, 0.0, None))

        return pessoas, alertas
