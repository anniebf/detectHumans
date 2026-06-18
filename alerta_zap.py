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
import requests  # <-- ADICIONADO PARA ENVIAR A MENSAGEM DO WHATSAPP

# Importações da nova Interface Gráfica
import customtkinter as ctk
from PIL import Image, ImageTk

load_dotenv()

# Configuração visual do CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

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

# --- CONFIGURAÇÃO DE COOLDOWN DO WHATSAPP ---
ultimo_envio_whatsapp = 0.0
WHATSAPP_COOLDOWN = 10.0  # Tempo em segundos para evitar spam

cliente_openai = None
if CONFIRMAR_QUEDA_AI:
    try:
        cliente_openai = OpenAI()
    except Exception:
        cliente_openai = None


def enviar_alerta_whatsapp_safe():
    """Envia o alerta com imagem respeitando o cooldown de 10 segundos"""
    global ultimo_envio_whatsapp
    agora = time.time()
    
    if agora - ultimo_envio_whatsapp >= WHATSAPP_COOLDOWN:
        ultimo_envio_whatsapp = agora
        
        # Captura o frame exato do momento do alerta de forma segura
        frame_alerta = None
        with frame_lock:
            if latest_frame is not None:
                frame_alerta = latest_frame.copy()
        
        # Converte o frame para Base64
        foto_b64 = None
        if frame_alerta is not None:
            sucesso, buffer = cv2.imencode(".jpg", frame_alerta, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if sucesso:
                foto_b64 = base64.b64encode(buffer).decode("utf-8")

        # Dispara a requisição em segundo plano para não travar a câmera
        threading.Thread(
            target=executar_requisicao_whatsapp, 
            args=(foto_b64,), 
            daemon=True
        ).start()


def executar_requisicao_whatsapp(foto_b64):
    url = "http://localhost:3000/client/sendMessage/ABCD"
    
    # Se conseguimos gerar o Base64 da foto, usamos a estrutura de media do controller
    if foto_b64:
        payload = {
            "chatId": "556593323330@c.us",
            "contentType": "string",
            "content": "🚨 ALERTA: Foi detectada uma possível queda no ambiente monitorado!",
            "options": {
                "media": {
                    "mimetype": "image/jpeg",
                    "data": foto_b64
                }
            }
        }
    else:
        # Fallback caso a conversão da imagem falhe por algum motivo
        payload = {
            "chatId": "556593323330@c.us",
            "contentType": "string",
            "content": "🚨 ALERTA: Queda detectada! (Falha ao capturar imagem da câmera)"
        }

    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
        if response.status_code == 200:
            print("[WhatsApp] Alerta com foto disparado com sucesso.")
        else:
            print(f"[WhatsApp] Erro ao enviar mensagem ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"[WhatsApp] Falha de conexão com a API: {e}")

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


# --- Seletor de Câmera Gráfico Moderno ---
def escolher_camera_gui():
    escolha = {"index": 0}
    dialog = ctk.CTk()
    dialog.title("Selecionar Câmera")
    dialog.geometry("320x180")
    dialog.resizable(False, False)
    
    label = ctk.CTkLabel(dialog, text="Escolha a câmera de monitoramento:", font=("Arial", 14, "bold"))
    label.pack(pady=15)
    
    def btn_click(idx):
        escolha["index"] = idx
        dialog.destroy()
        
    btn_0 = ctk.CTkButton(dialog, text="0 - Webcam Integrada", command=lambda: btn_click(0))
    btn_0.pack(pady=5)
    btn_1 = ctk.CTkButton(dialog, text="1 - Webcam Externa", command=lambda: btn_click(1))
    btn_1.pack(pady=5)
    
    dialog.mainloop()
    return escolha["index"]

indice_escolhido = escolher_camera_gui()
cap = abrir_webcam(indice_escolhido)

if cap is None:
    outro_indice = 1 if indice_escolhido == 0 else 0
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
ai_buscando = False 

latest_frame = None
annotated_frame = None
running = True

frame_lock = threading.Lock()


def capture_thread_fn(cap_obj):
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
    global status_ai, alerta_disparado, ai_buscando
    resultado_ai, erro_ai = confirmar_risco_com_openai(frames_para_enviar)
    
    if resultado_ai is not None:
        risco_ai = int(resultado_ai.get('risco', 0))
        categoria_ai = resultado_ai.get('categoria', '')
        is_queda = resultado_ai.get('queda', False)
        
        if is_queda or risco_ai >= 40 or categoria_ai in ['queda', 'risco_alto']:
            alerta_disparado = True
            status_ai = f'PERIGO: CHÃO DETECTADO ({risco_ai}%)'
            enviar_alerta_whatsapp_safe()  # <-- DISPARA O WHATSAPP COM FILTRO DE SPAM
        else:
            status_ai = f'Ignorado ({risco_ai}%) - Seguro'
            alerta_disparado = False
    else:
        alerta_disparado = True
        status_ai = f"Alerta Local (IA Offline/Erro)"
        enviar_alerta_whatsapp_safe()  # <-- DISPARA O WHATSAPP COM FILTRO DE SPAM EM CASO DE ERRO DA IA
        
    ai_buscando = False


def inference_thread_fn():
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
                if (agora - ultima_chamada_ai >= TEMPO_COOLDOWN_AI) and not ai_buscando:
                    ultima_chamada_ai = agora
                    ai_buscando = True
                    status_ai = "ANALISANDO IMAGEM..."
                    
                    if len(frames_suspeita_ai) == 0:
                        frame_b64 = preparar_crop_risco(frm, melhor_box)
                        if frame_b64:
                            frames_suspeita_ai.append(frame_b64)
                    
                    dados_envio = list(frames_suspeita_ai)
                    threading.Thread(target=async_openai_worker, args=(dados_envio,), daemon=True).start()
            else:
                alerta_disparado = True
                status_ai = "ALERTA: CORPO NO CHÃO"
                enviar_alerta_whatsapp_safe()  # <-- DISPARA O WHATSAPP COM FILTRO DE SPAM (Caso a IA esteja desligada)

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
        '{"risco": int(0-100), "queda": bool, "motivo": "string descritiva corta", "categoria": "queda"/"risco_alto"/"seguro"}'
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

# --- CONSTRUÇÃO DA INTERFACE GRÁFICA MODERNA (CustomTkinter) ---
root = ctk.CTk()
root.title("Guardião IA - Visão Computacional")
root.geometry("1024x600")
root.minsize(800, 480)

# Estrutura de Layout em Grid Responsivo
root.grid_columnconfigure(0, weight=1)  # Painel de Vídeo expande
root.grid_columnconfigure(1, weight=0)  # HUD fixo na direita
root.grid_rowconfigure(0, weight=1)

# Frame da Esquerda (Vídeo)
video_container = ctk.CTkFrame(root, corner_radius=0, fg_color="#0a0a0a")
video_container.grid(row=0, column=0, sticky="nsew")

video_label = ctk.CTkLabel(video_container, text="")
video_label.pack(fill="both", expand=True, padx=10, pady=10)

# Frame da Direita (HUD Painel de Controle)
hud_panel = ctk.CTkFrame(root, width=300, corner_radius=0, fg_color="#18181b")
hud_panel.grid(row=0, column=1, sticky="nsew")
hud_panel.grid_propagate(False)

# Elementos Textuais do HUD Gráfico
lbl_titulo = ctk.CTkLabel(hud_panel, text="SISTEMA GUARDIÃO IA", font=("Arial", 16, "bold"), text_color="#f4f4f5")
lbl_titulo.pack(pady=(30, 5), padx=20, anchor="w")

sub_linha = ctk.CTkFrame(hud_panel, height=2, fg_color="#3f3f46")
sub_linha.pack(fill="x", padx=20, pady=(0, 25))

# Card de Status Dinâmico
status_card = ctk.CTkFrame(hud_panel, height=60, corner_radius=8, fg_color="#27272a")
status_card.pack(fill="x", padx=20, pady=10)
lbl_status = ctk.CTkLabel(status_card, text="SISTEMA NORMAL", font=("Arial", 14, "bold"), text_color="#10b981")
lbl_status.pack(expand=True)

# Informações Auxiliares
lbl_pessoas = ctk.CTkLabel(hud_panel, text="Pessoas em cena: 0", font=("Arial", 13), text_color="#a1a1aa")
lbl_pessoas.pack(pady=8, padx=20, anchor="w")

lbl_hardware = ctk.CTkLabel(hud_panel, text=f"Hardware: {DEVICE} (0 FPS)", font=("Arial", 13), text_color="#a1a1aa")
lbl_hardware.pack(pady=8, padx=20, anchor="w")

lbl_ia = ctk.CTkLabel(hud_panel, text=f"IA: {status_ai}", font=("Arial", 12), text_color="#71717a", wraplength=260, justify="left")
lbl_ia.pack(pady=15, padx=20, anchor="w")

# Alerta de Emergência Visual Redesenhado
alert_box = ctk.CTkFrame(hud_panel, height=70, corner_radius=8, fg_color="#7f1d1d")
lbl_alert_text = ctk.CTkLabel(alert_box, text="🚨 ALERTA ENVIADO", font=("Arial", 14, "bold"), text_color="#fca5a5")
lbl_alert_text.pack(expand=True)

def fechar_aplicativo():
    global running
    running = False
    cap.release()
    root.quit()

btn_sair = ctk.CTkButton(hud_panel, text="Encerrar Monitor", fg_color="#3f3f46", hover_color="#52525b", command=fechar_aplicativo)
btn_sair.pack(side="bottom", pady=20, padx=20, fill="x")

# --- LOOP DE ATUALIZAÇÃO DA INTERFACE (Substitui o while antigo) ---
def update_gui_loop():
    global prev_time, annotated_frame, latest_frame, running, alerta_disparado, contador_pessoas, status_ai
    
    if not running:
        return

    with frame_lock:
        if annotated_frame is not None:
            frame_cv = annotated_frame.copy()
        else:
            frame_cv = latest_frame.copy() if latest_frame is not None else None

    if frame_cv is not None:
        # Medição de FPS
        current_time = time.time()
        fps = int(1 / (current_time - prev_time)) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Obter o tamanho atual do container para redimensionamento perfeito inteligente
        win_w = max(video_label.winfo_width(), 10)
        win_h = max(video_label.winfo_height(), 10)

        # Trata o aspecto mantendo a proporção correta
        img_h, img_w, _ = frame_cv.shape
        proporcao_original = img_w / img_h
        proporcao_janela = win_w / win_h

        if proporcao_janela > proporcao_original:
            novo_h = win_h
            novo_w = int(win_h * proporcao_original)
        else:
            novo_w = win_w
            novo_h = int(win_w / proporcao_original)

        # Redimensionamento suave livre de aliasing de câmeras antigas
        frame_resized = cv2.resize(frame_cv, (novo_w, novo_h), interpolation=cv2.INTER_LINEAR)
        
        # Converte BGR para RGB e joga no Tkinter
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        video_label.configure(image=img_tk)
        video_label.image = img_tk

        # Atualizações do Painel HUD Lateral Gráfico
        lbl_pessoas.configure(text=f"Pessoas em cena: {contador_pessoas}")
        lbl_hardware.configure(text=f"Hardware: {DEVICE} ({fps} FPS)")
        lbl_ia.configure(text=f"IA: {status_ai}")

        if alerta_disparado:
            status_card.configure(fg_color="#450a0a")
            lbl_status.configure(text="EMERGÊNCIA / QUEDA", text_color="#ef4444")
            if not alert_box.winfo_manager():
                alert_box.pack(fill="x", padx=20, pady=10, before=btn_sair)
        else:
            status_card.configure(fg_color="#14532d")
            lbl_status.configure(text="SISTEMA NORMAL", text_color="#4ade80")
            if alert_box.winfo_manager():
                alert_box.pack_forget()

    # Agenda a próxima execução do frame (~30 FPS)
    root.after(30, update_gui_loop)

# Dispara o loop interno do Tkinter
root.after(100, update_gui_loop)
root.protocol("WM_DELETE_WINDOW", fechar_aplicativo)
root.mainloop()