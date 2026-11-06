"""
main_pose.py - Monitor de quedas usando pose (MediaPipe) em vez do YOLO treinado.

LÓGICA GERAL
------------
  [Thread de captura]  lê a câmera o mais rápido possível e guarda só o quadro
                       mais recente (descarta atrasados -> baixa latência).
  [Loop principal]     pega o quadro mais recente
                       -> EstimadorPose (33 pontos por pessoa)
                       -> DetectorQueda (rastreia + máquina de estados)
                       -> desenha esqueleto/estado e dispara alertas.

Uso:
    python main_pose.py                      # webcam 0
    python main_pose.py --fonte 1            # webcam 1
    python main_pose.py --fonte rtsp://usuario:senha@ip:554/stream
    python main_pose.py --fonte video_queda.mp4   # teste com vídeo gravado
    python main_pose.py --sem-janela         # roda sem interface (servidor)

Tecla 'q' encerra.
"""

import argparse
import platform
import threading
import time

import cv2
from dotenv import load_dotenv

load_dotenv()  # carrega .env antes de criar as configs (elas leem variáveis de ambiente)

from sentinela_pose.alertas import NotificadorAlertas  # noqa: E402
from sentinela_pose.config import ConfigPose, ConfigQueda  # noqa: E402
from sentinela_pose.fall_detector import DetectorQueda, Estado, EstadoPessoa  # noqa: E402
from sentinela_pose.pontos import CONEXOES_ESQUELETO  # noqa: E402
from sentinela_pose.pose_estimator import EstimadorPose  # noqa: E402

LARGURA, ALTURA = 640, 360  # resolução de processamento: suficiente para pose e barata em CPU

# Cores BGR por estado.
CORES = {
    Estado.NORMAL: (80, 220, 80),
    Estado.NO_CHAO: (0, 200, 255),
    Estado.CAINDO: (0, 140, 255),
    Estado.ALERTA: (0, 0, 255),
}


class CapturaVideo:
    """Lê a fonte de vídeo em thread própria e mantém só o quadro mais recente."""

    def __init__(self, fonte: str):
        self.eh_arquivo = not fonte.isdigit() and not fonte.lower().startswith(("rtsp", "http"))
        if fonte.isdigit():
            # No Windows, CAP_DSHOW abre webcams bem mais rápido.
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(int(fonte), backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)
        else:
            self.cap = cv2.VideoCapture(fonte)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {fonte}")

        self.fps_arquivo = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._quadro = None
        self._lock = threading.Lock()
        self.rodando = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.rodando:
            ok, quadro = self.cap.read()
            if not ok:
                if self.eh_arquivo:
                    self.rodando = False  # fim do vídeo
                time.sleep(0.01)
                continue
            quadro = cv2.resize(quadro, (LARGURA, ALTURA))
            with self._lock:
                self._quadro = quadro
            if self.eh_arquivo:
                # Para vídeo gravado, respeita o FPS original (senão "acelera" o tempo da queda).
                time.sleep(1.0 / self.fps_arquivo)

    def ler(self):
        with self._lock:
            quadro, self._quadro = self._quadro, None  # consome: não reprocessa o mesmo quadro
        return quadro

    def fechar(self):
        self.rodando = False
        self.cap.release()


def desenhar(quadro, pessoas: list[EstadoPessoa], fps: float):
    """Desenha esqueleto, ID, estado e ângulo do tronco de cada pessoa."""
    for p in pessoas:
        cor = CORES[p.estado]
        if p.pose is not None:
            pts = p.pose.pontos
            for a, b in CONEXOES_ESQUELETO:
                if pts[a, 2] > 0.3 and pts[b, 2] > 0.3:
                    cv2.line(quadro, (int(pts[a, 0]), int(pts[a, 1])), (int(pts[b, 0]), int(pts[b, 1])), cor, 2)
            x, y = int(pts[:, 0].min()), int(max(pts[:, 1].min() - 10, 15))
        else:
            x, y = 10, 60  # pessoa fora de vista mas ainda em estado de risco
        texto = f"ID{p.id_pessoa} {p.estado.value.upper()} {p.angulo_tronco:.0f}deg"
        cv2.putText(quadro, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

    cv2.putText(quadro, f"{fps:.0f} FPS", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if any(p.estado == Estado.ALERTA for p in pessoas):
        cv2.rectangle(quadro, (0, 0), (quadro.shape[1] - 1, quadro.shape[0] - 1), (0, 0, 255), 6)
    return quadro


def main():
    parser = argparse.ArgumentParser(description="Sentinela - detecção de quedas por pose")
    parser.add_argument("--fonte", default="0", help="índice da webcam, URL RTSP ou caminho de vídeo")
    parser.add_argument("--sem-janela", action="store_true", help="não abre janela (modo servidor)")
    args = parser.parse_args()

    cfg_pose, cfg_queda = ConfigPose(), ConfigQueda()
    captura = CapturaVideo(args.fonte)
    estimador = EstimadorPose(cfg_pose)
    detector = DetectorQueda(LARGURA, ALTURA, cfg_queda, cfg_pose)
    notificador = NotificadorAlertas()

    t_anterior = time.time()
    fps = 0.0
    try:
        while captura.rodando:
            quadro = captura.ler()
            if quadro is None:
                time.sleep(0.005)
                continue

            agora = time.time()
            poses = estimador.processar(quadro, agora)
            pessoas, alertas = detector.processar(poses, agora)

            # FPS suavizado (média móvel exponencial) para exibição.
            dt = agora - t_anterior
            t_anterior = agora
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            tela = desenhar(quadro.copy(), pessoas, fps)
            for evento in alertas:
                notificador.notificar(evento, tela)

            if not args.sem_janela:
                cv2.imshow("Sentinela - Pose", tela)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        captura.fechar()
        estimador.fechar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
