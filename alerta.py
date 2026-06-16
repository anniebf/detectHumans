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
INFERENCE_CONF = 0.15       # Reduzido de 0.2 para 0.15 para garantir que veja a pessoa no chão
INFERENCE_IOU = 0.40        # Melhorado o descarte de caixas sobrepostas
MAX_DETECTIONS = 5          # Reduzido de 10 para 5 para economizar CPU

device_env = os.getenv("DEVICE")
if device_env is None:
    DEVICE = "cpu"
else:
    DEVICE = int(device_env) if device_env.isdigit() else device_env

USE_HALF = os.getenv("USE_HALF", "0") == "1"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USAR_OPENAI = os.getenv("OPENAI_CONFIRMAR_QUEDA", "1") == "1"
CONFIRMAR_QUEDA_AI = USAR_OPENAI and bool(os.getenv("OPENAI_API_KEY"))

# LIMIARES ADAPTADOS PARA CÂMERA ALTA EM DIAGONAIS
SUSPEITA_FRAMES_MIN = 2     # Reage mais rápido
LIMIAR_QUEDA_HORZ = 1.15    # Ajustado de 0.95 para 1.15 para pegar corpos deitados em perspectiva
LIMIAR_EM_PE = 1.45
LIMIAR_TRANSICAO = 1.20

TEMPO_MAX_SUSPEITA = 1.5    # Menor tempo esperando antes de chamar a IA
TEMPO_COOLDOWN_AI = 3.0
MARGEM_CROP = 0.15
AI_SAMPLE_SECONDS = 0.5
AI_JANELA_FRAMES = 3
contador_pessoas = 0

cliente_openai = None
if CONFIRMAR_QUEDA_AI:
    try:
        cliente_openai = OpenAI()
    except Exception:
        cliente_openai = None


def configuring_capture(cap):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


def abrir_webcam(indice):
    tried = []
    indices_to_try = [indice] + [i for i in range(0, 8) if i != indice]

    for idx in indices_to_try:
        tried.append(idx)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            configuring_capture(cap)
            print(f"Webcam aberta com sucesso no indice {idx} (tentativas: {tried}).")
            return cap
        else:
            try:
                cap.release()
            except Exception:
                pass
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
    raise RuntimeError("Nao foi possivel abrir nenhuma camera.")

try:
    model.fuse()
except Exception:
    pass

# Variáveis globais de controle
tempo_inicio_queda = 0
alerta_disparado = False
prev_time = time.time()
suspeita_queda_frames = 0
ultima_chamada_ai = 0.0
status_ai = "IA aguardando"
frames_suspeita_ai = deque(maxlen=AI_JANELA_FRAMES)
ultimo_sample_ai = 0.0

# Linha de montagem de Threads sem gargalo de Lock
latest_frame = None
annotated_frame = None
running = True

frame_lock = threading.Lock()


def capture_thread_fn(cap_obj):
    """Thread otimizada: Limpa o buffer da câmera agressivamente para impedir travamentos"""
    global latest_frame, running
    while running and cap_obj.isOpened():
        # Esvazia buffers antigos remanescentes na fila da webcam
        cap_obj.grab() 
        ok, frm = cap_obj.retrieve()
        if not ok:
            time.sleep(0.01)
            continue
        
        with frame_lock:
            latest_frame = frm


def inference_thread_fn():
    """Thread assíncrona: Executa a inferência YOLO de maneira cadenciada sem congelar a UI"""
    global annotated_frame, running, ultima_chamada_ai, tempo_inicio_queda, alerta_disparado
    global suspeita_queda_frames, ultimo_sample_ai, status_ai, contador_pessoas
    
    worker_model = model
    infer_interval = 0.15  # ~7 Hz para manter precisão sem explodir CPU
    
    while running:
        start = time.time()
        
        with frame_lock:
            if latest_frame is None:
                frm = None
            else:
                frm = latest_frame  # Referência direta sem o gargalo do .copy()

        if frm is None:
            time.sleep(0.02)
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

        tela, queda_detectada_neste_frame, cp_count, melhor_box, postura_melhor = drawing_engine(frm, res)
        contador_pessoas = cp_count

        posture_state = "desconhecida"
        if melhor_box is not None:
            if postura_melhor in ["horizontal", "transicao"]:
                posture_state = "horizontal"
            else:
                posture_state = "em_pe"

        # Lógica de captura de amostras para envio à IA externa
        if posture_state == "horizontal":
            suspeita_queda_frames += 1
            agora = time.time()
            if agora - ultimo_sample_ai >= AI_SAMPLE_SECONDS:
                ultimo_sample_ai = agora
                frame_b64 = preparar_crop_risco(frm, melhor_box)
                if frame_b64:
                    frames_suspeita_ai.append(frame_b64)
        else:
            if suspeita_queda_frames > 0:
                suspeita_queda_frames -= 1
            if len(frames_suspeita_ai) > 0 and suspeita_queda_frames == 0:
                frames_suspeita_ai.popleft()

        # Condição de perigo iminente validada localmente
        risco_local = (posture_state == "horizontal") and (suspeita_queda_frames >= SUSPEITA_FRAMES_MIN)

        if risco_local:
            if tempo_inicio_queda == 0:
                tempo_inicio_queda = time.time()
            else:
                tempo_passado = time.time() - tempo_inicio_queda
                if tempo_passado > TEMPO_MAX_SUSPEITA:
                    if CONFIRMAR_QUEDA_AI and cliente_openai is not None and len(frames_suspeita_ai) >= 1:
                        if time.time() - ultima_chamada_ai >= TEMPO_COOLDOWN_AI:
                            ultima_chamada_ai = time.time()
                            
                            status_ai = "Analisando com GPT..."
                            resultado_ai, erro_ai = confirmar_risco_com_openai(list(frames_suspeita_ai))
                            
                            if resultado_ai is not None:
                                risco_ai = int(resultado_ai.get('risco', 0))
                                categoria_ai = resultado_ai.get('categoria', '')
                                is_queda = resultado_ai.get('queda', False)
                                
                                # Critério de emergência inteligente baseado no seu ambiente
                                if is_queda or risco_ai >= 45 or categoria_ai in ['queda', 'risco_alto']:
                                    alerta_disparado = True
                                    status_ai = f'PERIGO: CHÃO DETECTADO ({risco_ai}%)'
                                else:
                                    status_ai = f'Descartado ({risco_ai}%) - {resultado_ai.get("motivo", "")[:15]}'
                                    tempo_inicio_queda = 0
                                    suspeita_queda_frames = 0
                                    frames_suspeita_ai.clear()
                                    alerta_disparado = False
                            else:
                                status_ai = erro_ai or 'IA Offline'
                    else:
                        if not CONFIRMAR_QUEDA_AI:
                            alerta_disparado = True
        else:
            if posture_state == "em_pe":
                tempo_inicio_queda = 0
                alerta_disparado = False

        with frame_lock:
            annotated_frame = tela

        elapsed = time.time() - start
        to_sleep = max(0.0, infer_interval - elapsed)
        time.sleep(to_sleep)


def expandir_bbox(x1, y1, x2, y2, largura_frame, altura_frame, margem=MARGEM_CROP):
    largura = x2 - x1
    altura = y2 - y1
    margem_x = int(largura * margem)
    margem_y = int(altura * margem)
    return max(0, x1 - margem_x), max(0, y1 - margem_y), min(largura_frame, x2 + margem_x), min(altura_frame, y2 + margem_y)


def preparar_crop_risco(frame, box):
    try:
        xyxy = box.xyxy[0].cpu().numpy()
        bx1, by1, bx2, by2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        h_frame, w_frame = frame.shape[:2]
        cx1, cy1, cx2, cy2 = expandir_bbox(bx1, by1, bx2, by2, w_frame, h_frame)
        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            return None

        if crop.shape[1] > 400:
            crop = cv2.resize(crop, (400, int(crop.shape[0] * (400 / crop.shape[1]))), interpolation=cv2.INTER_AREA)

        sucesso, buffer = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not sucesso:
            return None
        return base64.b64encode(buffer).decode("utf-8")
    except Exception:
        return None


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


def confirmar_risco_com_openai(frames_base64):
    if cliente_openai is None:
        return None, "OpenAI fora de servico"

    # Prompt reescrito de forma impositiva para evitar respostas negligentes da IA
    prompt = (
        "IMPORTANTE: Você monitora o chão de um quarto. Se o ser humano estocado na imagem "
        "estiver deitado diretamente sobre o chão, azulejos ou tapetes, responda como QUEDA e risco alto, "
        "mesmo que pareça brincando, deitado de bruços, descansando ou posando. Corpo totalmente horizontalizado "
        "no nível inferior é emergência no chão.\n"
        "Retorne OBRIGATORIAMENTE um objeto em JSON com as chaves exatas:\n"
        '{"risco": int(0-100), "queda": bool, "motivo": "string curta", "categoria": "queda"/"risco_alto"/"seguro"}'
    )

    imagens = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}} for f in frames_base64]

    try:
        resposta = cliente_openai.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Você é um operador de emergência e segurança residencial rígido."},
                {"role": "user", "content": [{"type": "text", "text": prompt}, *imagens]},
            ],
        )
        return json.loads(resposta.choices[0].message.content), None
    except Exception as erro:
        return None, f"Erro JSON: {str(erro)[:15]}"


def drawing_engine(frame, result):
    tela = frame.copy()
    queda_detectada = False
    
    if result.boxes is None or len(result.boxes) == 0:
        return tela, False, 0, None, "sem_pessoa"

    # Seleção inteligente baseada na maior área em tela
    melhor_box = None
    maior_area = -1
    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        if area > maior_area:
            maior_area = area
            melhor_box = box

    contador_p = len(result.boxes)
    postura_melhor = "indefinida"

    for box in result.boxes:
        postura, bx1, by1, bx2, by2, proporcao = classificar_postura_box(box)
        eh_principal = (box is melhor_box)

        if eh_principal:
            postura_melhor = postura
            cor = (0, 0, 255) if postura == "horizontal" else (0, 255, 255)
            espessura = 3
            
            if postura == "horizontal":
                queda_detectada = True
                cv2.putText(tela, "ALVO NO CHAO", (bx1, max(by1 - 12, 20)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(tela, f"MONITORANDO ({proporcao:.1f})", (bx1, max(by1 - 12, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
        else:
            cor = (120, 120, 120)
            espessura = 1

        cv2.rectangle(tela, (bx1, by1), (bx2, by2), cor, espessura)

    return tela, queda_detectada, contador_p, melhor_box, postura_melhor


# Inicialização das Threads de Processamento Pararelo
threading.Thread(target=capture_thread_fn, args=(cap,), daemon=True).start()
threading.Thread(target=inference_thread_fn, daemon=True).start()

print("Serviço ativo. Renderizando painel...")

FPS_DESEJADO = 30
tempo_por_frame = 1.0 / FPS_DESEJADO

# LOOP PRINCIPAL: Leitura limpa e renderização imediata do HUD
while running:
    start_loop = time.time()

    # Consome preferencialmente o frame processado pela Inteligência Artificial
    with frame_lock:
        if annotated_frame is not None:
            display_final = annotated_frame
        else:
            display_final = latest_frame

    if display_final is None:
        display_final = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8)

    h, w, _ = display_final.shape
    cor_hud_principal = (0, 0, 255) if alerta_disparado else (0, 255, 170)

    # Construção do Painel Lateral Técnico (HUD)
    hud_width = 320
    interface = cv2.copyMakeBorder(display_final, 0, 0, 0, hud_width, cv2.BORDER_CONSTANT, value=(15, 15, 15))
    start_x = w

    current_time = time.time()
    fps = int(1 / (current_time - prev_time)) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # Elementos Estéticos e Dados do Painel
    cv2.line(interface, (start_x, 0), (start_x, h), cor_hud_principal, 2)
    cv2.putText(interface, "MONITOR DE SEGURANCA", (start_x + 20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    cv2.line(interface, (start_x + 20, 55), (start_x + 250, 55), (60, 60, 60), 1)
    
    status_texto = "EMERGENCIA / QUEDA" if alerta_disparado else "SISTEMA NORMAL"
    cv2.putText(interface, f"STATUS: {status_texto}", (start_x + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_hud_principal, 1)
    cv2.putText(interface, f"PESSOAS EM CENA: {contador_pessoas}", (start_x + 20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(interface, f"HARDWARE: {DEVICE} ({fps} FPS)", (start_x + 20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(interface, f"IA: {status_ai}", (start_x + 20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if alerta_disparado:
        cv2.rectangle(interface, (start_x + 15, 240), (start_x + 300, 295), (0, 0, 180), -1)
        cv2.putText(interface, f"ALERTA ENVIADO", (start_x + 45, 275), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

    cv2.putText(interface, "Aperte 'Q' para fechar", (start_x + 20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    
    cv2.imshow("Guardiao IA - Visao Computacional", interface)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

    # Controlador de Latência para impedir acúmulo em máquinas limitadas
    tempo_gasto = time.time() - start_loop
    tempo_restante = tempo_por_frame - tempo_gasto
    if tempo_restante > 0:
        time.sleep(tempo_restante)

cap.release()
cv2.destroyAllWindows()