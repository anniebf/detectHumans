import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
DATA_YAML = ROOT_DIR / "dataset" / "data.yaml"
BASE_WEIGHTS = ROOT_DIR / "yolov8n.pt"
TRAINED_WEIGHTS = ROOT_DIR / "runs" / "detect" / "detect_human" / "ex1" / "weights" / "best.pt"


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def run_train(args: argparse.Namespace) -> None:
    model = YOLO(str(args.base_weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        augment=args.augment,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
    )

    if args.run_val:
        metrics = model.val(data=str(args.data), imgsz=args.imgsz, batch=args.batch, device=args.device)
        print(metrics)


def run_val(args: argparse.Namespace) -> None:
    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.data), imgsz=args.imgsz, batch=args.batch, device=args.device)
    print(metrics)


def run_webcam(args: argparse.Namespace) -> None:
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Arquivo de pesos nao encontrado: {weights_path}. "
            "Treine o modelo antes ou passe --weights com um caminho valido."
        )

    model = YOLO(str(weights_path))
    model.predict(
        source=parse_source(args.source),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        show=True,
        save=args.save,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        classes=args.classes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treino, validacao e teste em webcam com YOLOv8.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Treinar modelo")
    train_parser.add_argument("--data", type=Path, default=DATA_YAML, help="Caminho do arquivo data.yaml")
    train_parser.add_argument("--base-weights", type=Path, default=BASE_WEIGHTS, help="Pesos base para treino")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--device", default="cpu", help="Use 'cpu' ou indice da GPU, ex: 0")
    train_parser.add_argument("--workers", type=int, default=1)
    train_parser.add_argument("--lr0", type=float, default=0.01)
    train_parser.add_argument("--augment", action="store_true", default=True)
    train_parser.add_argument("--no-augment", dest="augment", action="store_false")
    train_parser.add_argument("--project", type=Path, default=ROOT_DIR / "runs" / "detect")
    train_parser.add_argument("--name", default="detect_human/ex1")
    train_parser.add_argument("--exist-ok", action="store_true", default=True)
    train_parser.add_argument("--run-val", action="store_true", default=True)
    train_parser.add_argument("--no-run-val", dest="run_val", action="store_false")
    train_parser.set_defaults(func=run_train)

    val_parser = subparsers.add_parser("val", help="Validar modelo treinado")
    val_parser.add_argument("--weights", type=Path, default=TRAINED_WEIGHTS, help="Pesos do modelo treinado")
    val_parser.add_argument("--data", type=Path, default=DATA_YAML, help="Caminho do arquivo data.yaml")
    val_parser.add_argument("--imgsz", type=int, default=640)
    val_parser.add_argument("--batch", type=int, default=8)
    val_parser.add_argument("--device", default="cpu", help="Use 'cpu' ou indice da GPU, ex: 0")
    val_parser.set_defaults(func=run_val)

    webcam_parser = subparsers.add_parser("webcam", help="Testar deteccao em webcam")
    webcam_parser.add_argument("--weights", type=Path, default=TRAINED_WEIGHTS, help="Pesos do modelo treinado")
    webcam_parser.add_argument("--source", default="0", help="Indice da webcam (0, 1, ...) ou URL RTSP/HTTP")
    webcam_parser.add_argument("--imgsz", type=int, default=640)
    webcam_parser.add_argument("--conf", type=float, default=0.25, help="Confianca minima")
    webcam_parser.add_argument("--iou", type=float, default=0.45, help="Limiar de IoU para NMS")
    webcam_parser.add_argument("--device", default="cpu", help="Use 'cpu' ou indice da GPU, ex: 0")
    webcam_parser.add_argument("--save", action="store_true", help="Salvar video com anotacoes")
    webcam_parser.add_argument("--project", type=Path, default=ROOT_DIR / "runs" / "detect")
    webcam_parser.add_argument("--name", default="webcam_test")
    webcam_parser.add_argument("--classes", type=int, nargs="+", help="Lista de classes para filtrar")
    webcam_parser.set_defaults(func=run_webcam)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()