# sentinela_pose — detecção de quedas por pose (alternativa ao YOLO treinado)

## Por que mudar
| Problema no fluxo atual | Causa | Como a pose resolve |
|---|---|---|
| Pesado | `ultralytics` + `torch` CUDA (~2,5 GB) e modelo de **segmentação** | MediaPipe Pose *lite* (~3 MB), CPU, sem PyTorch |
| Não detecta | modelo treinado com 690 imagens e validado nas **mesmas** imagens do treino (métricas infladas) | modelo pré-treinado do Google em grande volume de dados |
| Falso positivo/negativo | decisão pela proporção da caixa (`altura/largura >= 1.05`), que muda com a perspectiva da câmera | ângulo do tronco (ombro↔quadril) + velocidade de descida + tempo no chão |

## Como funciona
1. `pose_estimator.py`: 33 pontos do corpo por pessoa.
2. `fall_detector.py`:
   - **ângulo do tronco**: 0° em pé, 90° deitado;
   - **velocidade de descida do quadril** normalizada pelo tamanho do tronco;
   - **máquina de estados**: `NORMAL → CAINDO/NO_CHAO → ALERTA`, com tolerância a perdas curtas da pose.
3. `alertas.py`: salva imagem em `alertas_queda/` e envia WhatsApp (mesma API local já usada).

## Rodar
```bash
pip install -r requirements_pose.txt
python main_pose.py                    # webcam
python main_pose.py --fonte video.mp4  # testar com vídeo de queda
python -m pytest -v                    # testes da lógica (sem câmera)
```
Na 1ª execução o modelo é baixado para `modelos/`.

## Calibração
Ajuste os limiares no `.env` (ver `.env.exemplo_pose`). Use vídeos públicos de queda
(ex.: UR Fall Detection Dataset, Le2i) com `--fonte` para medir acertos antes de ir para a ILPI.

## Limitações
- Câmera muito alta (vista de cima) reduz a variação de ângulo: prefira câmeras a 2–2,5 m em diagonal.
- Corpo deitado e parcialmente oculto (cama, cadeira) pode perder a pose; a máquina de estados mantém a contagem por até `QUEDA_TEMPO_PERDA_MAX`.
- Muitas pessoas cruzando: o rastreador é simples (vizinho mais próximo).
