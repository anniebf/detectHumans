import cv2
import time
from ultralytics import YOLO

# 1. Carregar o seu modelo treinado (ajuste o caminho se a pasta mudar)
# Quando seu treino terminar, mude para o caminho do 'best.pt'
model = YOLO(r"C:\DetectHumans\runs\segment\train3\weights\best.pt")

# 2. Iniciar a Webcam (0 é o índice da câmera padrão do notebook)
cap = cv2.VideoCapture(0)

# Configurar resolução para HD (opcional, dependendo da sua webcam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = 0

print("Aperte 'Q' no teclado para fechar a câmera.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Erro ao acessar a webcam.")
        break

    # Altura e largura originais da captura da câmera
    h, w, _ = frame.shape

    # 3. Rodar o modelo de segmentação no frame atual
    results = model(frame, stream=True, conf=0.4) # conf=0.4 ignora detecções fracas

    contador_pessoas = 0

    for r in results:
        # Se houver máscaras/polígonos detectados
        if r.masks is not None:
            contador_pessoas = len(r.masks)
            
            # Desenha as máscaras coloridas geradas pelo YOLO de forma suave
            # (O parâmetro alpha controla a transparência da máscara)
            frame = r.plot(conf=True, line_width=2, font_size=1)

    # 4. Criando a Interface Bonita (HUD Lateral)
    # Criamos uma barra lateral escura para exibir as estatísticas do sistema
    hud_width = 320
    # Expande o tamanho da tela para caber o menu lateral
    interface = cv2.copyMakeBorder(frame, 0, 0, 0, hud_width, cv2.BORDER_CONSTANT, value=(20, 20, 20))

    # Desenhar linhas divisórias e detalhes estéticos (Design Futurista)
    start_x = w
    cv2.rectangle(interface, (start_x, 0), (start_x + hud_width, h), (30, 30, 30), -1)
    cv2.line(interface, (start_x, 0), (start_x, h), (0, 255, 170), 2) # Linha Neon

    # Calcular o FPS real da aplicação
    current_time = time.time()
    fps = int(1 / (current_time - prev_time)) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # Adicionar textos customizados na barra lateral
    cv2.putText(interface, "SISTEMA DE VISÃO IA", (start_x + 20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 170), 2)
    cv2.line(interface, (start_x + 20, 55), (start_x + 200, 55), (100, 100, 100), 1)

    cv2.putText(interface, f"STATUS: ATIVO", (start_x + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(interface, f"FPS: {fps}", (start_x + 20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    # Altera a cor do contador dependendo se há pessoas na tela ou não
    cor_alerta = (0, 0, 255) if contador_pessoas > 0 else (0, 255, 0)
    cv2.putText(interface, f"ALVO [PESSOA]: {contador_pessoas}", (start_x + 20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_alerta, 2)

    # Nota de rodapé da interface
    cv2.putText(interface, "Pressione 'Q' para sair", (start_x + 20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # 5. Exibir a tela final montada
    cv2.imshow("Monitoramento Inteligente - IA", interface)

    # Se pressionar a tecla 'q', fecha a janela da câmera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()