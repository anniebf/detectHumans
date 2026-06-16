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
INFERENCE_CONF = 0.15       # Sensibilidade alta para não perder o contorno no chão
INFERENCE_IOU = 0.45        
MAX_DETECTIONS = 5          

device_env = os.getenv("DEVICE")
if device_env is None:
    DEVICE = "cpu"
else:
    DEVICE = int(device_env) if device_env.isdigit() else device_env

USE_HALF = os.getenv("USE_HALF", "0") == "1"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USAR_OPENAI = os.getenv("OPENAI_CONFIRMAR_QUEDA", "1") == "1"
CONFIRMAR_QUEDA_AI = USAR_OPENAI and bool(os.getenv("OPENAI_API_KEY"))

# --- AJUSTE DE LIMIARES AGRESSIVOS PARA PERSPECTIVA ---
SUSPEITA_FRAMES_MIN = 2     # Apenas 2 frames horizontais/transição ativam o alerta
LIMIAR_EM_PE = 1.40         # Se for menor que 1.40, já deixa de ser considerado "Em Pé" normal
TEMPO_COOLDOWN_AI = 4.0     # Cooldown entre requisições de IA
MARGEM_CROP = 0.15
AI_SAMPLE_SECONDS = 0.4
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
status_ai = "IA aguardando detecção"
frames_suspeita_ai = deque(maxlen=AI_JANELA_FRAMES)
ultimo_sample_ai = 0.0

# Flag de controle para evitar chamadas duplicadas concorrentes à API
ai_buscando = False 

# Gerenciamento de Memória entre Threads
latest_frame = None
annotated_frame = None
running = True

frame_lock = threading.Lock()


def capture_thread_fn(cap_obj):
    """Limpa o buffer de hardware da câmera para eliminar travamentos e engasgos"""
    global latest_frame, running
    while running and cap_obj.isOpened():
        cap_obj.grab() 
        ok, frm = cap_obj.retrieve()
        if not ok:
            time.sleep(0.01)
            continue
        
        with frame_lock:
            latest_frame = frm


def async_openai_worker(frames_para_enviar):
    """Executa a chamada HTTP pesada em background sem congelar o loop de vídeo"""
    global status_ai, alerta_disparado, ai_buscando
    
    resultado_ai, erro_ai = confirmar_risco_com_openai(frames_para_enviar)
    
    if resultado_ai is not None:
        risco_ai = int(resultado_ai.get('risco', 0))
        categoria_ai = resultado_ai.get('categoria', '')
        is_queda = resultado_ai.get('queda', False)
        
        if is_queda or risco_ai >= 40 or categoria_ai in ['queda', 'risco_alto']:
            alerta_disparado = True
            status_ai = f'PERIGO: CHÃO DETECTADO ({risco_ai}%)'
        else:
            status_ai = f'Ignorado ({risco_ai}%) - Seguro'
            alerta_disparado = False
    else:
        # Fallback local se a API falhar (Garante a segurança)
        alerta_disparado = True
        status_ai = f"Alerta Local (IA Offline/Erro)"
        
    ai_buscando = False


def inference_thread_fn():
    """Analisa os frames, calcula posturas e aciona emergência de forma imediata"""
    global annotated_frame, running, ultima_chamada_ai, tempo_inicio_queda, alerta_disparado
    global suspeita_queda_frames, ultimo_sample_ai, status_ai, contador_pessoas, ai_buscando
    
    worker_model = model
    infer_interval = 0.12  
    
    while running:
        start = time.time()
        
        with frame_lock:
            if latest_frame is None:
                frm = None
            else:
                frm = latest_frame.copy() if latest_frame is not None else None

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

        tela, cp_count, melhor_box, postura_melhor = drawing_engine(frm, res)
        contador_pessoas = cp_count

        if melhor_box is not None and postura_melhor != "em_pe":
            posture_state = "risco_chao"
        elif melhor_box is not None and postura_melhor == "em_pe":
            posture_state = "em_pe"
        else:
            posture_state = "sem_alvo"

        # SISTEMA DE DISPARO DE ALERTA
        if posture_state == "risco_chao":
            suspeita_queda_frames += 1
            
            agora = time.time()
            if agora - ultimo_sample_ai >= AI_SAMPLE_SECONDS:
                ultimo_sample_ai = agora
                frame_b64 = preparar_crop_risco(frm, melhor_box)
                if frame_b64:
                    frames_suspeita_ai.append(frame_b64)
        else:
            if posture_state == "em_pe":
                suspeita_queda_frames = 0
                frames_suspeita_ai.clear()
                tempo_inicio_queda = 0
                alerta_disparado = False
                status_ai = "Monitorando... Tudo Normal"

        if posture_state == "risco_chao" and suspeita_queda_frames >= SUSPEITA_FRAMES_MIN:
            if tempo_inicio_queda == 0:
                tempo_inicio_queda = time.time()

            if CONFIRMAR_QUEDA_AI and cliente_openai is not None:
                agora = time.time()
                # Só dispara a Thread em background se respeitar o cooldown e não houver outra ativa
                if (agora - ultima_chamada_ai >= TEMPO_COOLDOWN_AI) and not ai_buscando:
                    ultima_chamada_ai = agora
                    ai_buscando = True
                    status_ai = "ANALISANDO IMAGEM..."
                    
                    if len(frames_suspeita_ai) == 0:
                        frame_b64 = preparar_crop_risco(frm, melhor_box)
                        if frame_b64:
                            frames_suspeita_ai.append(frame_b64)
                    
                    # Cria e dispara a thread isolada (A mágica da otimização está aqui)
                    dados_envio = list(frames_suspeita_ai)
                    threading.Thread(target=async_openai_worker, args=(dados_envio,), daemon=True).start()
            else:
                alerta_disparado = True
                status_ai = "ALERTA: CORPO NO CHÃO"

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

    if abrir_webcam and proporcao >= LIMIAR_EM_PE:
        return "em_pe", bx1, by1, bx2, by2, proporcao
    
    return "horizontal", bx1, by1, bx2, by2, proporcao


def confirmar_risco_com_openai(frames_base64):
    if cliente_openai is None:
        return None, "OpenAI Offline"

    prompt = (
        "IMPORTANTE DE SEGURANÇA: Avalie se a pessoa detectada nas imagens da câmera está caída ou deitada no chão, "
        "independente se parece que está brincando, descansando, engatinhando ou dormindo. "
        "Estar posicionado no nível do chão ou tapete do quarto configura situação de risco alto para este sistema.\n"
        "Retorne estritamente um objeto JSON com as chaves:\n"
        '{"risco": int(0-100), "queda": bool, "motivo": "string descritiva curta", "categoria": "queda"/"risco_alto"/"seguro"}'
    )

    imagens = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}} for f in frames_base64]

    try:
        resposta = cliente_openai.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Você é um monitor automático focado em segurança de pessoas vulneráveis no chão."},
                {"role": "user", "content": [{"type": "text", "text": prompt}, *imagens]},
            ],
        )
        return json.loads(resposta.choices[0].message.content), None
    except Exception as erro:
        return None, f"Erro API: {str(erro)[:15]}"


def drawing_engine(frame, result):
    tela = frame.copy()
    
    if result.boxes is None or len(result.boxes) == 0:
        return tela, 0, None, "sem_pessoa"

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
            cor = (0, 0, 255) if postura != "em_pe" else (0, 255, 170)
            espessura = 3
            
            if postura != "em_pe":
                cv2.putText(tela, "ALVO NO CHAO / RISCO", (bx1, max(by1 - 12, 20)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(tela, f"EM PE ({proporcao:.1f})", (bx1, max(by1 - 12, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
        else:
            cor = (100, 100, 100)
            espessura = 1

        cv2.rectangle(tela, (bx1, by1), (bx2, by2), cor, espessura)

    return tela, contador_p, melhor_box, postura_melhor


# Inicialização das Threads de Processamento Paralelo
threading.Thread(target=capture_thread_fn, args=(cap,), daemon=True).start()
threading.Thread(target=inference_thread_fn, daemon=True).start()

print("Serviço ativo. Renderizando painel...")

FPS_DESEJADO = 30
tempo_por_frame = 1.0 / FPS_DESEJADO

# LOOP PRINCIPAL: Sem concorrência, focado apenas na interface fluida
while running:
    start_loop = time.time()

    with frame_lock:
        if annotated_frame is not None:
            display_final = annotated_frame.copy()
        else:
            display_final = latest_frame.copy() if latest_frame is not None else None

    if display_final is None:
        display_final = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8)

    h, w, _ = display_final.shape
    cor_hud_principal = (0, 0, 255) if alerta_disparado else (0, 255, 170)

    hud_width = 320
    interface = cv2.copyMakeBorder(display_final, 0, 0, 0, hud_width, cv2.BORDER_CONSTANT, value=(15, 15, 15))
    start_x = w

    current_time = time.time()
    fps = int(1 / (current_time - prev_time)) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # Elementos do HUD Lateral
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

    tempo_gasto = time.time() - start_loop
    tempo_restante = tempo_por_frame - tempo_gasto
    if tempo_restante > 0:
        time.sleep(tempo_restante)

cap.release()
cv2.destroyAllWindows()