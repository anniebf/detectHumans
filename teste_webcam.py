import cv2
import time
import torch
from ultralytics import YOLO

# 1. Carregar o seu modelo treinado (ajuste o caminho se a pasta mudar)
# Quando seu treino terminar, mude para o caminho do 'best.pt'
model = YOLO(r"C:\DetectHumans\runs\segment\train3\weights\best.pt")

CAPTURE_WIDTH = 960
CAPTURE_HEIGHT = 540
INFERENCE_IMGSZ = 416
INFERENCE_CONF = 0.18
INFERENCE_IOU = 0.35
INFERENCE_EVERY_N_FRAMES = 2
MAX_DETECTIONS = 10
DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()

def configurar_capture(cap):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


def abrir_webcam(indice):
    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    if cap.isOpened():
        configurar_capture(cap)
        print(f"Webcam aberta com sucesso no indice {indice}.")
        return cap

    cap.release()
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


# 2. Iniciar a webcam escolhida pelo usuario
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

prev_time = 0
frame_index = 0
ultimo_contador_pessoas = 0
ultimo_frame_processado = None

print("Aperte 'Q' no teclado para fechar a câmera.")


def desenhar_resultado(frame, result):
    tela = frame.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return tela, 0

    contador = 0
    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = [int(valor) for valor in xyxy]
        contador += 1

        cv2.rectangle(tela, (x1, y1), (x2, y2), (0, 255, 170), 2)
        cv2.putText(
            tela,
            "PESSOA",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 170),
            2,
        )

    return tela, contador

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Erro ao acessar a webcam.")
        break

    # Altura e largura originais da captura da câmera
    h, w, _ = frame.shape

    frame_index += 1

    # 3. Rodar o modelo em uma cadência menor para manter o FPS mais estável.
    if frame_index % INFERENCE_EVERY_N_FRAMES == 0 or ultimo_frame_processado is None:
        result = model.predict(
            source=frame,
            imgsz=INFERENCE_IMGSZ,
            conf=INFERENCE_CONF,
            iou=INFERENCE_IOU,
            max_det=MAX_DETECTIONS,
            verbose=False,
            device=DEVICE,
            half=USE_HALF,
            classes=[0],
        )[0]
        ultimo_frame_processado, ultimo_contador_pessoas = desenhar_resultado(frame, result)

    contador_pessoas = ultimo_contador_pessoas
    frame = ultimo_frame_processado if ultimo_frame_processado is not None else frame

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