from ultralytics import YOLO

# 1. Carregar um modelo pré-treinado (recomendado)
# Você pode escolher entre: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
modelo = YOLO('yolov8n.pt')   # 'n' é o modelo nano (mais leve e rápido)

# Se quiser treinar do zero (sem pesos pré-treinados), use:
# modelo = YOLO('yolov8n.yaml')

# 2. Treinar o modelo
resultados = modelo.train(
    data='C:/YoloProjIntegrador/dataset/data.yaml',      # arquivo de configuração do dataset
    epochs=100,               # número de épocas
    imgsz=640,                # tamanho da imagem de entrada
    batch=8,                  # tamanho do batch (ajuste conforme sua GPU)
    device='cpu',                  # dispositivo: 0 para GPU, 'cpu' para CPU
    workers=1,                 # número de threads para carregar dados
    lr0=0.01,                  # learning rate inicial
    augment=True,              # aplicar data augmentation
    project='detect_human', # pasta onde salvar os resultados
    name='ex1',                # subpasta com o nome do experimento
    exist_ok=True               # sobrescreve se a pasta já existir
)

# 3. (Opcional) Validar o modelo treinado
metricas = modelo.val()
print(metricas)

# 4. (Opcional) Exportar para outros formatos (ONNX, TensorRT, etc.)
# modelo.export(format='onnx')