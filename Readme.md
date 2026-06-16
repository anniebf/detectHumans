# SENTINELA - Detecção de Quedas com Visão Computacional

> **Projeto Integrador** - Sistema inteligente de monitoramento de segurança para detecção automática de quedas de pessoas idosas usando YOLO v8 e análise com IA.

---

## Criadores

- **Hianny**
- **Thalia** 
- **Igor** 
- **Claverson** 

**Instituição:** Univag  
**Data:** 2026

---

## Objetivo do Projeto

Desenvolver um sistema automático de **detecção de quedas em tempo real** que:

✅ Detecta pessoas caídas ou deitadas no chão usando segmentação YOLO v8  
✅ Valida detecções com análise de IA (OpenAI GPT-4)  
✅ Fornece interface gráfica interativa para monitoramento  
✅ Gera alertas imediatos de emergência  
✅ Funciona com webcam em tempo real (30+ FPS)

**Aplicações práticas:**
- Monitoramento de idosos em residências
- Segurança em ambientes hospitalares
- Detecção de acidentes em espaços públicos
- Assistência a pessoas com mobilidade reduzida

---

## Dataset

O dataset foi anotado usando **CVAT (Computer Vision Annotation Tool)** com:

- **926 imagens** coletadas e anotadas manualmente
- **2 classes de segmentação:**
  - `pessoa_caida`: pessoa deitada ou caída no chão
  - `pessoa_em_pe`: pessoa em posição normal (em pé)
- **Formato:** Polígonos de segmentação (máscaras) exportadas em formato YOLO
- **Divisão:** ~80% treino, ~20% validação

Estrutura dos dados:
```
dataset2/
├── data.yaml              # Configuração do dataset
├── images/
│   └── train/            # 741 imagens de treino
└── labels/
    └── train/            # 741 máscaras de segmentação (.txt)
```

---

## Como Funciona o Sistema

### 1. **Captura e Processamento de Frames**
```python
# Thread dedicada para captura otimizada
- Captura frames da webcam em tempo real (640x360)
- Evita buffering para baixa latência
- Sincroniza com thread de inferência
```

### 2. **Detecção com YOLO v8 Segmentation**
```python
model = YOLO("runs/segment/train3/weights/best.pt")
resultado = model.predict(
    source=frame,
    imgsz=320,           # Tamanho otimizado para velocidade
    conf=0.15,           # Sensibilidade alta para não perder detecções
    iou=0.45,
    max_det=5,
    device="cpu"         # Ou GPU
)
```

**Processos de classificação:**
- **Posição em Pé:** altura/largura da caixa ≥ 1.40
- **Risco de Queda:** altura/largura da caixa < 1.40
- **Limiar ativação:** 2 frames consecutivos em risco

### 3. **Validação com OpenAI GPT-4**
Quando uma queda é suspeita:

```python
def confirmar_risco_com_openai(frames_base64):
    # Envia 3 frames codificados em base64 para análise
    prompt = """
    Avalie se a pessoa está CAÍDA ou DEITADA no chão.
    Retorne JSON com: risco (0-100), queda (bool), categoria
    """
    resposta = cliente_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é especialista em segurança..."},
            {"role": "user", "content": [{"type": "text", ...}, imagens...]}
        ]
    )
```

**Resposta esperada:**
```json
{
  "risco": 92,
  "queda": true,
  "motivo": "Pessoa claramente deitada no chão",
  "categoria": "queda"
}
```

### 4. **Interface Gráfica em Tempo Real**
Interface moderna com **CustomTkinter** mostrando:

-  Vídeo ao vivo com anotações
-  Número de pessoas detectadas
-  FPS e hardware utilizado
-  Status da análise de IA
-  Alerta visual em caso de emergência

---

##  Processo de Treinamento do YOLO

### Configuração

O arquivo [train.py](train.py) realiza o treinamento supervisionado:

```python
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  # Nano - rápido e leve

model.train(
    data="dataset2/data.yaml",   # Arquivo YAML com paths
    epochs=100,                   # 100 ciclos completos
    imgsz=640,                    # Tamanho padrão de treino
    batch=16,                     # 16 imagens por lote
    workers=4,                    # 4 threads de processamento
    device='cpu',                 # Ou GPU (ex: device=0)
)
```

### Arquivo de Configuração (data.yaml)

```yaml
path: /path/to/dataset2
train: images/train
val: images/val

nc: 2  # 2 classes
names: ['pessoa_caida', 'pessoa_em_pe']
```

### Métricas de Treinamento

Após treinamento com 926 imagens, esperamos:

| Métrica | Valor Esperado |
|---------|----------------|
| **mAP50** | ~0.88-0.92 |
| **mAP50-95** | ~0.78-0.85 |
| **Precisão** | ~0.89 |
| **Recall** | ~0.86 |
| **FPS (CPU)** | ~25-30 |
| **FPS (GPU)** | ~100+ |

### Exemplo de Curva de Treinamento

```
Epoch 1/100:   Loss: 2.45  | mAP50: 0.12
Epoch 10/100:  Loss: 0.89  | mAP50: 0.45
Epoch 30/100:  Loss: 0.34  | mAP50: 0.71
Epoch 50/100:  Loss: 0.18  | mAP50: 0.82
Epoch 100/100: Loss: 0.08  | mAP50: 0.90 ✓
```

Os pesos treinados são salvos em:
```
runs/segment/train3/weights/
├── best.pt    # Melhor modelo (validação)
└── last.pt    # Último checkpoint
```

---

##  Estrutura do Projeto

```
DetectHumans/
├── alerta.py                    #  Script principal com detecção + GUI
├── train.py                     #  Script de treinamento YOLO
├── listar_cameras.py            #  Utilitário para listar câmeras
├── teste_webcam.py              #  Teste de webcam
├── requirements.txt             #  Dependências Python
├── yolov8n.pt                   #  Modelo pré-treinado
├── dataset/                     #  Dataset antigo usando somente 1 pessoas nas fotos de treino
│   ├── data.yaml
│   ├── images/
│   │   ├── train/              # ~741 imagens
│   │   └── val/                # ~185 imagens
│   └── labels/                 # Anotações YOLO
├── dataset2/                    # Dataset atual usando fotos mais variadas e com mais de 1 pessoa
│   ├── data.yaml
│   ├── train.py
│   ├── images/train/
│   └── labels/train/
├── runs/
│   ├── detect/                 # Resultados de detecção
│   └── segment/
│       └── train3/
│           └── weights/
│               ├── best.pt     # ⭐ Modelo final usado
│               └── last.pt
└── alertas_queda/              #  Alertas salvos automaticamente
```

---

##  Como Executar

### 1. Instalação de Dependências

```bash
# Criar ambiente virtual (recomendado)
python -m venv .venv

# Ativar (Windows)
.venv\Scripts\activate

# Ou (Linux/Mac)
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Treinar o Modelo (Caso for fazer do zero)

```bash
python train.py
```

O treinamento levará entre 1 a 2 dias para treinar e salvará os pesos em `runs/segment/train3/weights/best.pt`.

### 3. Executar o Sistema de Detecção

```bash
python alerta.py
```

**Ao iniciar:**
1. Selecione a câmera desejada (0 = integrada, 1 = externa)
2. Aguarde a conexão com a webcam
3. Sistema começa a monitorar automaticamente

**Indicadores Visuais:**
- 🟢 **Verde:** Pessoa em pé, sem risco
- 🔴 **Vermelho:** Queda detectada, alerta ativado
- 🟡 **Amarelo:** Análise com IA em andamento

---

## 🔧 Configurações Importantes

Arquivo `alerta.py` - Constantes ajustáveis:

```python
# Qualidade da câmera
CAPTURE_WIDTH = 640             # Largura dos frames
CAPTURE_HEIGHT = 360            # Altura dos frames

# Detecção YOLO
INFERENCE_IMGSZ = 320           # Tamanho otimizado para velocidade
INFERENCE_CONF = 0.15           # Confiança mínima
INFERENCE_IOU = 0.45            # Supressão de não-máximo

# Limiares de queda
SUSPEITA_FRAMES_MIN = 2         # Frames para ativar alerta
LIMIAR_EM_PE = 1.40             # Proporção altura/largura

# OpenAI
OPENAI_API_KEY = "sua_chave_aqui"      # .env file
OPENAI_MODEL = "gpt-4o-mini"
TEMPO_COOLDOWN_AI = 4.0         # Segundos entre requisições
```

### Variáveis de Ambiente (.env)

```
DEVICE=0                        # 0=GPU, cpu=CPU
USE_HALF=0                      # Half precision (0=não, 1=sim)
OPENAI_API_KEY=sk-...          # Sua chave OpenAI
OPENAI_CONFIRMAR_QUEDA=1       # Usar IA para validar (1=sim)
OPENAI_MODEL=gpt-4o-mini       # Modelo a usar
```

---

##  Exemplos de Funcionamento

### Exemplo 1: Detecção em Tempo Real

```
Frame 1: Pessoa em pé → Postura: "em_pe" → Status: ✅ Normal
Frame 2: Pessoa em pé → Postura: "em_pe" → Status: ✅ Normal
Frame 3: Pessoa caída → Postura: "risco_chao" → Contador: 1
Frame 4: Pessoa caída → Postura: "risco_chao" → Contador: 2 ⚠️ ALERTA!

Enviando para IA...
Resposta: {"risco": 95, "queda": true, "categoria": "queda"}

🚨 EMERGÊNCIA DETECTADA - ALERTA ENVIADO
```

### Exemplo 2: Métrica de Treinamento (CSV dos Resultados)

```
epoch  box_loss  cls_loss  seg_loss  mAP50  mAP50-95  FPS
0      2.456    0.789    1.234    0.145  0.089    22
10     0.876    0.234    0.567    0.456  0.312    24
50     0.234    0.087    0.156    0.821  0.734    25
100    0.087    0.034    0.067    0.901  0.824    25
```

### Exemplo 3: Saída do Modelo na GUI

```
┌─────────────────────────────┐
│   GUARDIÃO IA MONITOR       │
├─────────────────────────────┤
│ Status: EMERGÊNCIA / QUEDA  │  🔴 (Vermelho)
│ Pessoas: 1                  │
│ FPS: 28                     │
│ Hardware: CPU               │
│ IA: PERIGO: CHÃO (92%)      │
├─────────────────────────────┤
│ ALERTA ENVIADO              │
└─────────────────────────────┘
```

---

## Troubleshooting

| Problema | Solução |
|----------|---------|
| **Câmera não abre** | Verifique o índice (0 ou 1), ou use `listar_cameras.py` |
| **FPS baixo (< 10)** | Reduza `INFERENCE_IMGSZ` para 256, use GPU |
| **Muitos falsos positivos** | Aumente `INFERENCE_CONF` para 0.25-0.30 |
| **OpenAI timeout** | Verifique chave API, aumente `TEMPO_COOLDOWN_AI` |
| **Out of memory (GPU)** | Reduza `batch` size no treino ou use CPU |

---

## Dependências

Veja `requirements.txt`:

```
opencv-python==4.8.0
ultralytics==8.0.0
torch==2.1.0
torchvision==0.16.0
customtkinter==5.2.0
pillow==10.0.0
openai==1.3.0
python-dotenv==1.0.0
numpy==1.24.3
```

---

## Referências

- **YOLO v8 Docs:** https://docs.ultralytics.com/
- **OpenAI API:** https://platform.openai.com/docs/
- **CVAT Annotation:** https://www.cvat.ai/
- **CustomTkinter:** https://github.com/TomSchimansky/CustomTkinter

---

## Notas Importantes

-  **Privacidade:** Garanta consentimento para captura de vídeo
-  **Segurança:** Nunca compartilhe `OPENAI_API_KEY` em repositórios públicos
-  **Precisão:** Quanto mais dados anotados, melhor o desempenho
-  **Performance:** Use GPU para produção (100+ FPS vs 25 FPS em CPU)

---

##  Licença

Este projeto é desenvolvido para fins de aprendizado como parte de projeto integrador, caso a ideia seja reultilizda para outros meios o projeto original deve ser refenciada.

## Proximas Etapas

- aumentar e balancear o dataset;
- melhorar robustez em diferentes ambientes (iluminacao, angulos, oclusao);
- criar logica temporal para diferenciar postura normal de queda;
- integrar alertas para cenarios reais de monitoramento de idosos.

## Observacoes
Este repositorio representa uma etapa importante de base tecnica. O foco futuro e transformar a deteccao de pessoas em uma solucao pratica para **prevencao e resposta a acidentes**, especialmente no contexto de cuidado com idosos.
