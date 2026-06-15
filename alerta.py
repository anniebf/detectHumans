import base64
import json
import cv2
import time
import os
from collections import deque
from ultralytics import YOLO
from openai import OpenAI
from dotenv import load_dotenv
import threading
from typing import Optional
import numpy as np

load_dotenv()

# 1. Configurações Iniciais e Pastas de Alerta
ALERTA_DIR = "alertas_queda"
if not os.path.exists(ALERTA_DIR):
    os.makedirs(ALERTA_DIR)

# Carrega o seu modelo de segmentação treinado
model = YOLO(r"C:\DetectHumans\runs\segment\train3\weights\best.pt")

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 360
INFERENCE_IMGSZ = 320
INFERENCE_CONF = 0.2
INFERENCE_IOU = 0.35
MAX_DETECTIONS = 10
# Avoid importing torch at module import time on Windows (can hang due to platform checks).
# Default to CPU. To enable GPU, set environment variable DEVICE to the GPU index (e.g. 0)
# and set USE_HALF=1 in the environment if you know your GPU supports FP16.
device_env = os.getenv("DEVICE")
if device_env is None:
    DEVICE = "cpu"
else:
    DEVICE = int(device_env) if device_env.isdigit() else device_env

USE_HALF = os.getenv("USE_HALF", "0") == "1"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USAR_OPENAI = os.getenv("OPENAI_CONFIRMAR_QUEDA", "1") == "1"
CONFIRMAR_QUEDA_AI = USAR_OPENAI and bool(os.getenv("OPENAI_API_KEY"))
SUSPEITA_FRAMES_MIN = 3
LIMIAR_QUEDA_HORZ = 0.95
LIMIAR_EM_PE = 1.55
LIMIAR_TRANSICAO = 1.15
TEMPO_MAX_SUSPEITA = 2.2
TEMPO_COOLDOWN_AI = 2.5
MARGEM_CROP = 0.18
AI_SAMPLE_SECONDS = 0.7
AI_JANELA_FRAMES = 3
AI_GATILHO_FRAMES = 2
AI_RISCO_ALTO = 70
MARGEM_MOVIMENTO_PIXELS = 18

cliente_openai = None
if CONFIRMAR_QUEDA_AI:
    try:
        cliente_openai = OpenAI()
    except Exception:
        cliente_openai = None


def configurar_capture(cap):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


def abrir_webcam(indice):
    # Tenta abrir o indice preferido primeiro; se falhar, tenta outros indices comuns.
    tried = []
    indices_to_try = [indice] + [i for i in range(0, 8) if i != indice]

    for idx in indices_to_try:
        tried.append(idx)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            configurar_capture(cap)
            print(f"Webcam aberta com sucesso no indice {idx} (tentativas: {tried}).")
            return cap
        else:
            try:
                cap.release()
            except Exception:
                pass

    print(f"Falha ao abrir a webcam. Indices tentados: {tried}")
    return None


def escolher_camera():
    print("Escolha a camera:")
    print("  0 - Webcam do notebook/PC")
    print("  1 - Webcam externa")

    while True:
        opcao = input("Digite 0 ou 1: ").strip()
        if opcao in {"0", "1"}:
            return int(opcao)
        print("Opcao invalida. Digite 0 ou 1.")


indice_escolhido = escolher_camera()
cap = abrir_webcam(indice_escolhido)

if cap is None:
    outro_indice = 1 if indice_escolhido == 0 else 0
    print(f"Nao foi possivel abrir o indice {indice_escolhido}. Tentando o indice {outro_indice}...")
    cap = abrir_webcam(outro_indice)

if cap is None:
    raise RuntimeError(
        "Nao foi possivel abrir a camera escolhida nem a alternativa. Verifique se a webcam esta conectada e "
        "se o indice selecionado esta correto."
    )

try:
    model.fuse()
except Exception:
    pass

# Variáveis para controle do tempo de queda (evitar alarmes falsos)
tempo_inicio_queda = 0
alerta_disparado = False
tempo_cooldown_print = 0 # Evita tirar 30 prints por segundo durante a queda
prev_time = 0
frames_sem_pessoa = 0
suspeita_queda_frames = 0
postura_anterior = "desconhecida"
ultima_chamada_ai = 0.0
status_ai = "IA desativada"
frames_suspeita_ai = deque(maxlen=AI_JANELA_FRAMES)
ultimo_sample_ai = 0.0
historico_postura = deque(maxlen=AI_JANELA_FRAMES)

print("Sistema de Monitoramento Iniciado. Pressione 'Q' para sair.")

# Threading globals
latest_frame = None
latest_frame_lock = threading.Lock()
annotated_frame = None
annotated_frame_lock = threading.Lock()
inference_result = None
inference_lock = threading.Lock()
running = True


def load_preferred_model():
    """Load detection weights if provided via env var DETECTION_WEIGHTS, else use default model."""
    weights = os.getenv("DETECTION_WEIGHTS")
    if weights and os.path.exists(weights):
        print(f"Carregando pesos de deteccao: {weights}")
        return YOLO(str(weights))
    # fallback to existing model already loaded
    return model


worker_model: Optional[YOLO] = None
worker_model = load_preferred_model()


def capture_thread_fn(cap_obj):
    global latest_frame, running
    while running and cap_obj.isOpened():
        ok, frm = cap_obj.read()
        if not ok:
            time.sleep(0.01)
            continue
        with latest_frame_lock:
            latest_frame = frm.copy()
    print("Capture thread ending")


def inference_thread_fn():
    global annotated_frame, inference_result, running, ultima_chamada_ai, tempo_inicio_queda, alerta_disparado
    global suspeita_queda_frames, ultimo_sample_ai, status_ai
    
    infer_interval = 0.25  # 4 Hz
    while running:
        start = time.time()
        with latest_frame_lock:
            frm = None if latest_frame is None else latest_frame.copy()

        if frm is None:
            time.sleep(0.01)
            continue

        try:
            res = worker_model.predict(
                source=frm,
                imgsz=INFERENCE_IMGSZ,
                conf=INFERENCE_CONF,
                iou=INFERENCE_IOU,
                max_det=MAX_DETECTIONS,
                verbose=False,
                device=DEVICE,
                half=USE_HALF,
                classes=[0],
            )[0]
        except Exception as e:
            print("Erro na inferencia:", e)
            time.sleep(0.1)
            continue

        tela, queda_detectada_neste_frame, contador_pessoas, melhor_box, postura_melhor, geometria_melhor = desenhar_resultado(frm, res)

        if melhor_box is not None and postura_melhor == "horizontal":
            posture_state = "horizontal"
        elif melhor_box is not None and postura_melhor == "em_pe":
            posture_state = "em_pe"
        else:
            posture_state = "desconhecida"

        if geometria_melhor is not None:
            _, _, _, _, _, altura, centro_x, centro_y, proporcao = geometria_melhor
            historico_postura.append((centro_x, centro_y, altura, proporcao))

        # Gerenciamento da suspeita e coleta de frames para a IA externa
        if posture_state == "horizontal":
            suspeita_queda_frames += 1
            agora = time.time()
            if agora - ultimo_sample_ai >= AI_SAMPLE_SECONDS:
                ultimo_sample_ai = agora
                frame_b64 = preparar_crop_risco(frm, melhor_box)
                if frame_b64:
                    frames_suspeita_ai.append(frame_b64)
        else:
            # Em vez de dar .clear() imediato, vamos reduzindo gradativamente 
            # para não perder o histórico se o YOLO piscar em um frame
            if suspeita_queda_frames > 0:
                suspeita_queda_frames -= 1
            if len(frames_suspeita_ai) > 0 and suspeita_queda_frames == 0:
                frames_suspeita_ai.popleft()

        # Simplificação: se está horizontal e persistiu por alguns frames, aciona o gatilho local
        risco_local = (posture_state == "horizontal") and (suspeita_queda_frames >= SUSPEITA_FRAMES_MIN)

        if risco_local:
            if tempo_inicio_queda == 0:
                tempo_inicio_queda = time.time()
            else:
                tempo_passado = time.time() - tempo_inicio_queda
                # Se persistir deitado pelo tempo determinado
                if tempo_passado > TEMPO_MAX_SUSPEITA:
                    confirmar_ai = not CONFIRMAR_QUEDA_AI
                    
                    if CONFIRMAR_QUEDA_AI and cliente_openai is not None and len(frames_suspeita_ai) >= 1:
                        if time.time() - ultima_chamada_ai >= TEMPO_COOLDOWN_AI:
                            ultima_chamada_ai = time.time()
                            
                            contexto_texto = f"postura=horizontal; frames_coletados={len(frames_suspeita_ai)}"
                            
                            # Chamada para validar se a pessoa está caída no chão
                            resultado_ai, erro_ai = confirmar_risco_com_openai(list(frames_suspeita_ai), contexto_texto)
                            
                            if resultado_ai is not None:
                                risco_ai = int(resultado_ai.get('risco', 0))
                                categoria_ai = resultado_ai.get('categoria', '')
                                
                                if resultado_ai.get('queda', False) or categoria_ai in ['queda', 'risco_alto']:
                                    alerta_disparado = True
                                    status_ai = f'Confirmado chão ({risco_ai}%)'
                                else:
                                    status_ai = f'Descartado ({risco_ai}%) {categoria_ai}'
                                    tempo_inicio_queda = 0
                                    suspeita_queda_frames = 0
                                    frames_suspeita_ai.clear()
                                    alerta_disparado = False
                            else:
                                status_ai = erro_ai or 'IA indisponivel'
                    else:
                        if not CONFIRMAR_QUEDA_AI:
                            alerta_disparado = True
        else:
            # Se a pessoa levantou e sumiu o risco local por muito tempo, reseta o alarme
            if tempo_inicio_queda != 0 and (time.time() - tempo_inicio_queda > 5.0):
                tempo_inicio_queda = 0
                alerta_disparado = False

        with annotated_frame_lock:
            annotated_frame = tela

        globals()['contador_pessoas'] = contador_pessoas

        elapsed = time.time() - start
        to_sleep = max(0.0, infer_interval - elapsed)
        time.sleep(to_sleep)

def expandir_bbox(x1, y1, x2, y2, largura_frame, altura_frame, margem=MARGEM_CROP):
    largura = x2 - x1
    altura = y2 - y1
    margem_x = int(largura * margem)
    margem_y = int(altura * margem)

    novo_x1 = max(0, x1 - margem_x)
    novo_y1 = max(0, y1 - margem_y)
    novo_x2 = min(largura_frame, x2 + margem_x)
    novo_y2 = min(altura_frame, y2 + margem_y)
    return novo_x1, novo_y1, novo_x2, novo_y2


def preparar_crop_risco(frame, box):
    xyxy = box.xyxy[0].cpu().numpy()
    bx1, by1, bx2, by2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    h_frame, w_frame = frame.shape[:2]
    cx1, cy1, cx2, cy2 = expandir_bbox(bx1, by1, bx2, by2, w_frame, h_frame)
    crop = frame[cy1:cy2, cx1:cx2]

    if crop.size == 0:
        return None

    altura_crop, largura_crop = crop.shape[:2]
    max_largura = 512
    if largura_crop > max_largura:
        nova_altura = max(1, int(altura_crop * (max_largura / largura_crop)))
        crop = cv2.resize(crop, (max_largura, nova_altura), interpolation=cv2.INTER_AREA)

    sucesso, buffer = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not sucesso:
        return None

    return base64.b64encode(buffer).decode("utf-8")


def classificar_postura_box(box):
    xyxy = box.xyxy[0].cpu().numpy()
    bx1, by1, bx2, by2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    largura_caixa = max(1, bx2 - bx1)
    altura_caixa = max(1, by2 - by1)
    proporcao = altura_caixa / largura_caixa

    if proporcao >= LIMIAR_EM_PE:
        return "em_pe", bx1, by1, bx2, by2, proporcao

    if proporcao <= LIMIAR_QUEDA_HORZ:
        return "horizontal", bx1, by1, bx2, by2, proporcao

    return "transicao", bx1, by1, bx2, by2, proporcao


def selecionar_melhor_box(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    melhor_box = None
    melhor_pontuacao = -1.0

    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        bx1, by1, bx2, by2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        largura = max(1, bx2 - bx1)
        altura = max(1, by2 - by1)
        area = largura * altura
        confianca = float(box.conf[0].cpu().item()) if box.conf is not None else 0.0
        pontuacao = area * (0.5 + confianca)

        if pontuacao > melhor_pontuacao:
            melhor_pontuacao = pontuacao
            melhor_box = box

    return melhor_box


def extrair_geometria_box(box):
    xyxy = box.xyxy[0].cpu().numpy()
    bx1, by1, bx2, by2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    largura = max(1, bx2 - bx1)
    altura = max(1, by2 - by1)
    centro_x = bx1 + largura // 2
    centro_y = by1 + altura // 2
    proporcao = altura / largura
    return bx1, by1, bx2, by2, largura, altura, centro_x, centro_y, proporcao


def confirmar_risco_com_openai(frames_base64, contexto_texto):
    if cliente_openai is None:
        return None, "OpenAI nao configurada"

    prompt = (
        "Analise a imagem da camera de seguranca. Foque em identificar se o ser humano detectado "
        "esta deitado no chao, caído, desmaiado ou em uma situacao de vulnerabilidade (como chao de banheiro/sala). "
        "Responda APENAS em JSON valido com as chaves: "
        '"risco" (0 a 100), "queda" (boolean), "motivo" (string) e "categoria" (string). '
        "Se a pessoa estiver deitada no chao ou caída, marque 'queda': true e risco acima de 80. "
        "Se ela estiver apenas sentada em uma cadeira, em pe ou agachada normalmente trabalhando, marque 'queda': false. "
        f"Contexto: {contexto_texto}."
    )

    imagens = []
    for frame_base64 in frames_base64:
        imagens.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"},
            }
        )

    resposta = cliente_openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Você é um classificador visual de quedas para um sistema de segurança.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *imagens,
                ],
            },
        ],
    )

    texto = (resposta.choices[0].message.content or "").strip()
    if texto.startswith("```"):
        texto = texto.strip("`\n")

    try:
        dados = json.loads(texto)
        queda = bool(dados.get("queda", False))
        risco = int(dados.get("risco", 0))
        motivo = str(dados.get("motivo", ""))
        categoria = str(dados.get("categoria", ""))
        return {"queda": queda, "risco": risco, "motivo": motivo, "categoria": categoria}, None
    except Exception as erro:
        return None, f"Falha ao interpretar resposta da OpenAI: {erro}"


def desenhar_resultado(frame, result):
    tela = frame.copy()
    queda_detectada_neste_frame = False
    contador_pessoas = 0
    melhor_box = selecionar_melhor_box(result)

    if result.boxes is None or len(result.boxes) == 0:
        return tela, queda_detectada_neste_frame, contador_pessoas, None, "sem_pessoa", None

    contador_pessoas = len(result.boxes)

    # Draw only the main (best) box prominently; keep others minimal to reduce drawing cost.
    for box in result.boxes:
        postura, bx1, by1, bx2, by2, proporcao = classificar_postura_box(box)
        eh_melhor_box = melhor_box is not None and box is melhor_box

        if not eh_melhor_box:
            # thin light rectangles for secondary detections
            cv2.rectangle(tela, (bx1, by1), (bx2, by2), (100, 100, 100), 1)
            continue

        # prominent main box
        cor_caixa = (0, 255, 255)
        espessura = 3
        cv2.rectangle(tela, (bx1, by1), (bx2, by2), cor_caixa, espessura)
        cv2.putText(
            tela,
            f"PESSOA {proporcao:.2f}",
            (bx1, max(by1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            cor_caixa,
            2,
        )

        if postura == "horizontal":
            queda_detectada_neste_frame = True
            cv2.rectangle(tela, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
            cv2.putText(tela, "SUSPEITA DE QUEDA", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    postura_melhor = "indefinida"
    geometria_melhor = None
    if melhor_box is not None:
        postura_melhor, bx1, by1, bx2, by2, proporcao = classificar_postura_box(melhor_box)
        geometria_melhor = extrair_geometria_box(melhor_box)
        if postura_melhor == "horizontal":
            cv2.putText(tela, "ALVO PRINCIPAL: HORIZONTAL", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return tela, queda_detectada_neste_frame, contador_pessoas, melhor_box, postura_melhor, geometria_melhor

# Start capture and inference threads
cap_thread = threading.Thread(target=capture_thread_fn, args=(cap,), daemon=True)
cap_thread.start()
inf_thread = threading.Thread(target=inference_thread_fn, daemon=True)
inf_thread.start()

print("Threads de captura e inferencia iniciadas. Exibindo frames...")

while running:
    # get the latest annotated frame if available
    with annotated_frame_lock:
        display = None if annotated_frame is None else annotated_frame.copy()

    if display is None:
        with latest_frame_lock:
            display = None if latest_frame is None else latest_frame.copy()

    if display is None:
        display = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8)

    h, w, _ = display.shape

    # HUD values
    contador_pessoas = globals().get('contador_pessoas', 0)
    cor_hud_principal = (0, 0, 255) if globals().get('alerta_disparado', False) else (0, 255, 170)

    hud_width = 320
    interface = cv2.copyMakeBorder(display, 0, 0, 0, hud_width, cv2.BORDER_CONSTANT, value=(15, 15, 15))
    start_x = w

    # Cálculo de FPS no display
    current_time = time.time()
    fps = int(1 / (current_time - prev_time)) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    cv2.line(interface, (start_x, 0), (start_x, h), cor_hud_principal, 2)
    cv2.putText(interface, "MONITOR DE SEGURANCA", (start_x + 20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    cv2.line(interface, (start_x + 20, 55), (start_x + 250, 55), (60, 60, 60), 1)
    status_texto = "EMERGENCIA / QUEDA" if globals().get('alerta_disparado', False) else "SISTEMA NORMAL"
    cv2.putText(interface, f"STATUS: {status_texto}", (start_x + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_hud_principal, 1)
    cv2.putText(interface, f"PESSOAS EM CENA: {contador_pessoas}", (start_x + 20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(interface, f"HARDWARE: {DEVICE} ({fps} FPS)", (start_x + 20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(interface, f"IA: {globals().get('status_ai', '')}", (start_x + 20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    if globals().get('alerta_disparado', False):
        cv2.rectangle(interface, (start_x + 15, 230), (start_x + 300, 290), (0, 0, 180), -1)
        cv2.putText(interface, "ALERTA ENVIADO!", (start_x + 40, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(interface, "Aperte 'Q' para fechar", (start_x + 20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.imshow("Guardião IA - Visão Computacional", interface)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

# join threads and cleanup
cap_thread.join(timeout=1.0)
inf_thread.join(timeout=1.0)

cap.release()
cv2.destroyAllWindows()