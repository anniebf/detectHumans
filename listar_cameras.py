import cv2
import time

def listar_cameras():
    """Lista todas as câmeras disponíveis no sistema."""
    print("Escaneando câmeras disponíveis...")
    print("=" * 50)
    
    cameras_found = {}
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    backend_names = {cv2.CAP_DSHOW: "CAP_DSHOW", cv2.CAP_MSMF: "CAP_MSMF", cv2.CAP_ANY: "CAP_ANY"}
    
    for idx in range(10):
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    # Tenta obter propriedades
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    key = f"Índice {idx}"
                    if key not in cameras_found:
                        cameras_found[key] = []
                    cameras_found[key].append({
                        "backend": backend_names[backend],
                        "resolution": f"{width}x{height}"
                    })
                    
                    cap.release()
            except Exception:
                pass
    
    if not cameras_found:
        print("❌ Nenhuma câmera encontrada!")
        return
    
    print(f"✅ {len(cameras_found)} câmera(s) encontrada(s):\n")
    for key, backends_list in cameras_found.items():
        print(f"{key}:")
        for info in backends_list:
            print(f"  - Backend: {info['backend']}, Resolução: {info['resolution']}")
    
    print("\n" + "=" * 50)
    print("INSTRUÇÕES:")
    print("- Índice 0 geralmente é a webcam do notebook")
    print("- Índices mais altos (1, 2, 3...) são câmeras externas")
    print("=" * 50)

if __name__ == "__main__":
    listar_cameras()
