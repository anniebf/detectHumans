import cv2
import time
import os
from ultralytics import YOLO

# 1. Configurações Iniciais e Pastas de Alerta
ALERTA_DIR = "alertas_queda"
if not os.path.exists(ALERTA_DIR):
    os.makedirs(ALERTA_DIR)

# Carrega o seu modelo de segmentação treinado
model = YOLO(r"C:\DetectHumans\runs\segment\train3\weights\best.pt")

# Iniciar a Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variáveis para controle do tempo de queda (evitar alarmes falsos)
tempo_inicio_queda = 0
alerta_disparado = False
tempo_cooldown_print = 0 # Evita tirar 30 prints por segundo durante a queda
prev_time = 0

print("Sistema de Monitoramento Iniciado. Pressione 'Q' para sair.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    
    # Executa a detecção do YOLO no frame
    results = model(frame, stream=True, conf=0.4)
    
    queda_detectada_neste_frame = False
    contador_pessoas = 0

    for r in results:
        # Plota as máscaras de segmentação bonitas se houver detecção
        if r.masks is not None:
            contador_pessoas = len(r.masks)
            frame = r.plot(conf=False, line_width=2) # conf=False deixa o visual mais limpo

        # Analisa a caixa (bounding box) de cada pessoa para ver se caiu
        if r.boxes is not None:
            for box in r.boxes:
                # Pegar coordenadas da caixa: x1, y1 (topo esquerdo), x2, y2 (baixo direito)
                xyxy = box.xyxy[0].to('cpu').numpy()
                bx1, by1, bx2, by2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                
                largura_caixa = bx2 - bx1
                altura_caixa = by2 - by1
                
                # SE A LARGURA FOR MAIOR QUE A ALTURA = Alvo está na horizontal (deitado/caído)
                # Adicionamos uma altura mínima para evitar detectar objetos aleatórios no chão
                if largura_caixa > (altura_caixa * 1.2) and altura_caixa > 40:
                    queda_detectada_neste_frame = True
                    # Desenha um retângulo vermelho extra piscando ao redor do acidentado
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                    cv2.putText(frame, "PERIGO: QUEDA!", (bx1, by1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 2. Lógica do Alarme (A pessoa precisa ficar caída por 1.5 segundos para validar)
    if queda_detectada_neste_frame:
        if tempo_inicio_queda == 0:
            tempo_inicio_queda = time.time() # Começa a contar o relógio
        else:
            tempo_passado = time.time() - tempo_inicio_queda
            if tempo_passado > 1.5: # Passou de 1.5 segundos caída? É uma queda real!
                alerta_disparado = True
                
                # Sistema de Print de Alerta (Só tira um print a cada 5 segundos de queda)
                if time.time() - tempo_cooldown_print > 5:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    nome_print = os.path.join(ALERTA_DIR, f"ALERTA_QUEDA_{timestamp}.jpg")
                    # Salva o frame atual com os desenhos e marcações na pasta
                    cv2.imwrite(nome_print, frame)
                    print(f"[ALERTA] Print de emergência salvo em: {nome_print}")
                    tempo_cooldown_print = time.time()
    else:
        # Se a pessoa levantou ou sumiu da câmera, limpa o contador do alarme
        tempo_inicio_queda = 0
        alerta_disparado = False

    # 3. Construção do HUD Lateral Avançado
    hud_width = 320
    interface = cv2.copyMakeBorder(frame, 0, 0, 0, hud_width, cv2.BORDER_CONSTANT, value=(15, 15, 15))
    start_x = w
    
    # Linha divisória de status
    cor_hud_principal = (0, 0, 255) if alerta_disparado else (0, 255, 170)
    cv2.line(interface, (start_x, 0), (start_x, h), cor_hud_principal, 2)

    # Cálculo de FPS
    current_time = time.time()
    fps = int(1 / (current_time - prev_time)) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # Textos do Painel Lateral
    cv2.putText(interface, "MONITOR DE SEGURANCA", (start_x + 20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    cv2.line(interface, (start_x + 20, 55), (start_x + 250, 55), (60, 60, 60), 1)

    status_texto = "EMERGENCIA / QUEDA" if alerta_disparado else "SISTEMA NORMAL"
    cv2.putText(interface, f"STATUS: {status_texto}", (start_x + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_hud_principal, 1)
    cv2.putText(interface, f"PESSOAS EM CENA: {contador_pessoas}", (start_x + 20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(interface, f"HARDWARE: CPU ({fps} FPS)", (start_x + 20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Se o alerta estiver ativo, cria uma tarja vermelha piscando na barra de informações
    if alerta_disparado:
        cv2.rectangle(interface, (start_x + 15, 230), (start_x + 300, 290), (0, 0, 180), -1)
        cv2.putText(interface, "ALERTA ENVIADO!", (start_x + 40, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(interface, "Aperte 'Q' para fechar", (start_x + 20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # Mostrar a tela final
    cv2.imshow("Guardião IA - Visão Computacional", interface)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()