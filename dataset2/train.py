import os
from ultralytics import YOLO

def treinar_modelo():
    # 1. Definir o caminho para o arquivo de configuração do seu dataset
    # O CVAT gera o arquivo 'data.yaml' dentro da pasta exportada.
    caminho_yaml = os.path.join(".", "data.yaml")
    
    if not os.path.exists(caminho_yaml):
        print(f"Erro: O arquivo {caminho_yaml} não foi encontrado!")
        print("Verifique se a pasta foi renomeada corretamente para 'dataset2' e se o 'data.yaml' está dentro dela.")
        return

    print("--- Iniciando o Treinamento de Segmentação ---")
    
    # 2. Carregar o modelo YOLO pré-treinado para Segmentação (versão Nano - rápida e leve)
    # Usamos o 'yolov8n-seg.pt' pois você anotou com polígonos/pontos.
    model = YOLO("yolov8n-seg.pt")

    # 3. Iniciar o treinamento
    model.train(
        data=caminho_yaml,   # Caminho do arquivo data.yaml do seu dataset2
        epochs=100,          # Número de épocas (iterações completas sobre o dataset)
        imgsz=640,           # Tamanho da imagem para o treino (padrão 640x640)
        batch=16,            # Quantidade de imagens processadas por vez (reduza para 8 ou 4 se der erro de memória)
        workers=4,           # Linhas de processamento simultâneo
        device='cpu',             # Usa a placa de vídeo (GPU) 0. Mude para device='cpu' se não tiver GPU dedicada
        
    )
    
    print("--- Treinamento Concluído com Sucesso! ---")

if __name__ == "__main__":
    treinar_modelo()