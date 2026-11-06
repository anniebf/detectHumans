"""
alertas.py - Envio de alertas quando o detector confirma uma queda.

Mantém o mesmo contrato do script antigo (alerta_zap_ajuste_frames.py):
POST para a API local de WhatsApp com a foto do momento em Base64.
Diferenças:
  - URL e chatId vêm do .env (não ficam fixos no código: número de telefone é dado pessoal/LGPD).
  - O envio roda em thread separada para não travar o vídeo.
  - Sempre salva a imagem do alerta em disco (evidência para o histórico).

No futuro (API REST do Sentinela), basta criar outra função `enviar_*`
que faça POST /eventos e registrá-la em `NotificadorAlertas`.
"""

import base64
import os
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import requests

from .fall_detector import EventoAlerta

_MENSAGENS = {
    "queda": "🚨 ALERTA: queda detectada no ambiente monitorado!",
    "pessoa_no_chao": "⚠️ ALERTA: pessoa no chão há alguns segundos no ambiente monitorado.",
}


class NotificadorAlertas:
    """Salva a imagem e dispara o WhatsApp respeitando um intervalo mínimo entre envios."""

    def __init__(self, pasta: str = "alertas_queda", cooldown_s: float | None = None):
        self.pasta = pasta
        os.makedirs(self.pasta, exist_ok=True)
        self.cooldown_s = cooldown_s if cooldown_s is not None else float(os.getenv("ALERTA_COOLDOWN", 30))
        self.url_whatsapp = os.getenv("WHATSAPP_URL")        # ex.: http://localhost:3000/client/sendMessage/ABCD
        self.chat_id = os.getenv("WHATSAPP_CHAT_ID")         # ex.: 5565XXXXXXXXX@c.us
        self._ultimo_envio = 0.0

    def notificar(self, evento: EventoAlerta, quadro: Optional[np.ndarray]) -> None:
        """Chamado pelo loop principal. Não bloqueia."""
        agora = time.time()
        caminho = self._salvar_imagem(evento, quadro)
        print(f"[ALERTA] {evento.tipo} | pessoa {evento.id_pessoa} | tronco {evento.angulo_tronco:.0f}° | {caminho}")

        if agora - self._ultimo_envio < self.cooldown_s:
            return  # evita spam enquanto a mesma situação persiste
        self._ultimo_envio = agora

        if self.url_whatsapp and self.chat_id:
            threading.Thread(target=self._enviar_whatsapp, args=(evento, quadro), daemon=True).start()

    def _salvar_imagem(self, evento: EventoAlerta, quadro: Optional[np.ndarray]) -> Optional[str]:
        if quadro is None:
            return None
        nome = f"{datetime.now():%Y%m%d_%H%M%S}_{evento.tipo}_p{evento.id_pessoa}.jpg"
        caminho = os.path.join(self.pasta, nome)
        cv2.imwrite(caminho, quadro)
        return caminho

    def _enviar_whatsapp(self, evento: EventoAlerta, quadro: Optional[np.ndarray]) -> None:
        payload = {"chatId": self.chat_id, "contentType": "string", "content": _MENSAGENS[evento.tipo]}
        if quadro is not None:
            ok, buffer = cv2.imencode(".jpg", quadro, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                payload["options"] = {"media": {"mimetype": "image/jpeg",
                                                "data": base64.b64encode(buffer).decode("utf-8")}}
        try:
            resposta = requests.post(self.url_whatsapp, json=payload, timeout=10)
            print(f"[WhatsApp] status {resposta.status_code}")
        except requests.RequestException as erro:
            print(f"[WhatsApp] falha ao enviar: {erro}")
