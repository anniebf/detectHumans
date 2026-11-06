"""
config.py - Parâmetros ajustáveis do detector de quedas por pose.

LÓGICA GERAL DO PROJETO (sentinela_pose)
----------------------------------------
Em vez de treinar um YOLO próprio para "achar a pessoa" e depois olhar a
proporção da caixa (o que falha com perspectiva de câmera), este módulo usa
ESTIMATIVA DE POSE (MediaPipe Pose Landmarker, roda em CPU, sem PyTorch).

O modelo devolve 33 pontos do corpo (ombros, quadris, joelhos...). Com eles:
  1. Medimos o ÂNGULO DO TRONCO (linha quadril -> ombro) em relação à vertical.
     Em pé ~0°, deitado ~90°. Isso independe do tamanho da pessoa na imagem.
  2. Medimos a VELOCIDADE DE DESCIDA do quadril, normalizada pelo tamanho do
     tronco (assim funciona perto ou longe da câmera).
  3. Uma MÁQUINA DE ESTADOS por pessoa decide:
       - queda brusca + deitado por X s  -> ALERTA "queda"
       - deitado por Y s (sem ver a queda) -> ALERTA "pessoa_no_chao"
       - voltou a ficar em pé por Z s      -> volta ao normal

Todos os limiares ficam aqui para facilitar a calibração em campo.
Cada valor pode ser sobrescrito por variável de ambiente (arquivo .env).
"""

import os
from dataclasses import dataclass, field


def _env_float(nome: str, padrao: float) -> float:
    """Lê um float do ambiente; se não existir ou for inválido, usa o padrão."""
    try:
        return float(os.getenv(nome, padrao))
    except ValueError:
        return padrao


def _env_int(nome: str, padrao: int) -> int:
    """Lê um int do ambiente; se não existir ou for inválido, usa o padrão."""
    try:
        return int(os.getenv(nome, padrao))
    except ValueError:
        return padrao


@dataclass
class ConfigPose:
    """Configuração do estimador de pose (MediaPipe)."""

    # Variante do modelo: "lite" (mais rápido), "full" ou "heavy" (mais preciso).
    variante_modelo: str = field(default_factory=lambda: os.getenv("POSE_MODELO", "lite"))
    # Pasta onde o arquivo .task é salvo (baixado automaticamente na 1ª execução).
    pasta_modelos: str = field(default_factory=lambda: os.getenv("POSE_PASTA_MODELOS", "modelos"))
    # Quantas pessoas o modelo procura por quadro (custo cresce com esse número).
    max_pessoas: int = field(default_factory=lambda: _env_int("POSE_MAX_PESSOAS", 3))
    # Confiança mínima para considerar que existe uma pessoa.
    conf_deteccao: float = field(default_factory=lambda: _env_float("POSE_CONF_DETECCAO", 0.4))
    conf_presenca: float = field(default_factory=lambda: _env_float("POSE_CONF_PRESENCA", 0.4))
    conf_rastreio: float = field(default_factory=lambda: _env_float("POSE_CONF_RASTREIO", 0.4))
    # Visibilidade mínima de um ponto (0..1) para ser usado nos cálculos.
    visibilidade_min: float = field(default_factory=lambda: _env_float("POSE_VISIBILIDADE_MIN", 0.3))


@dataclass
class ConfigQueda:
    """Limiares da lógica de queda. Tempos em segundos, ângulos em graus."""

    # Tronco acima deste ângulo (vs. vertical) = postura deitada/no chão.
    angulo_deitado: float = field(default_factory=lambda: _env_float("QUEDA_ANGULO_DEITADO", 60.0))
    # Tronco abaixo deste ângulo = em pé/sentado ereto (usado para "recuperou").
    angulo_em_pe: float = field(default_factory=lambda: _env_float("QUEDA_ANGULO_EM_PE", 35.0))
    # Descida do quadril, em "comprimentos de tronco por segundo", que caracteriza queda brusca.
    velocidade_queda: float = field(default_factory=lambda: _env_float("QUEDA_VELOCIDADE", 1.2))
    # Janela usada para medir a velocidade de descida.
    janela_velocidade: float = field(default_factory=lambda: _env_float("QUEDA_JANELA_VELOCIDADE", 0.5))
    # Uma queda brusca vale como "impacto" se a pessoa deitar até X s depois.
    janela_impacto: float = field(default_factory=lambda: _env_float("QUEDA_JANELA_IMPACTO", 2.0))
    # Tempo deitado, após impacto, para confirmar a queda.
    tempo_confirma_queda: float = field(default_factory=lambda: _env_float("QUEDA_TEMPO_CONFIRMA", 2.0))
    # Tempo deitado, sem impacto visto, para alertar "pessoa no chão".
    tempo_no_chao: float = field(default_factory=lambda: _env_float("QUEDA_TEMPO_NO_CHAO", 8.0))
    # Tempo em pé contínuo para considerar que a pessoa se recuperou.
    tempo_recuperacao: float = field(default_factory=lambda: _env_float("QUEDA_TEMPO_RECUPERACAO", 1.5))
    # Falhas curtas de detecção (oclusão, pose perdida) toleradas sem zerar contadores.
    tolerancia_falha: float = field(default_factory=lambda: _env_float("QUEDA_TOLERANCIA_FALHA", 1.0))
    # Após esse tempo sem ver a pessoa, o rastro dela é descartado.
    tempo_perda_max: float = field(default_factory=lambda: _env_float("QUEDA_TEMPO_PERDA_MAX", 10.0))
    # Distância máxima (fração da diagonal da imagem) para associar a mesma pessoa entre quadros.
    distancia_max_rastreio: float = field(default_factory=lambda: _env_float("QUEDA_DIST_MAX_RASTREIO", 0.15))
