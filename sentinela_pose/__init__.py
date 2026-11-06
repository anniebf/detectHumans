"""
sentinela_pose - Detecção de quedas por estimativa de pose (MediaPipe), leve e em CPU.

Módulos:
    config.py         -> limiares ajustáveis (também via .env)
    pontos.py         -> estrutura Pose e índices dos 33 pontos do corpo
    pose_estimator.py -> wrapper do MediaPipe Pose Landmarker
    fall_detector.py  -> rastreador + máquina de estados da queda (sem dependência do MediaPipe)
    alertas.py        -> salva imagem e envia WhatsApp
"""
