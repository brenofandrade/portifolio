# Detecção de Carros com YOLO

# Configuração do Ambiente de desenvolvimento
# python -m venv .venv
# .venv\Scripts\activate for Windows
# source .venv/bin/activate for Linux/Mac
# python -m pip install --upgrade pip
# pip install -r requirements.txt
# python deteccao_carros.py


from pathlib import Path
from typing import Dict, Set, Tuple, Optional

import cv2
from matplotlib.pyplot import box
import numpy as np
from ultralytics import YOLO

# ----------------------------- Configurações ----------------------------- #

VIDEO_PATH = Path(r"C:\Users\andra\repositorio\portifolio\video-analysis-yolo\data\viaduto.mp4")
MODEL_PATH = "yolov8n.pt"

# Classes COCO consideradas "veículo"
VEHICLE_CLASSES = {
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
}

CONFIDENCE = 0.35
LINE_Y_RATIO = 0.70  # posição da linha de contagem (80% da altura do frame)

# Sentido de contagem: "descendo", "subindo" ou "ambos"
DIRECAO_CONTAGEM = "ambos"

Center = Tuple[int, int]

# ----------------------------- Funções auxiliares ----------------------------- #

def load_model(model_path: str) -> YOLO:
    """
    Carrega o modelo YOLO a partir do caminho especificado.

    Args:
        model_path (str): Caminho para o arquivo do modelo YOLO.

    Returns:
        YOLO: Instância do modelo YOLO carregado.
    """
    try:
        model = YOLO(model_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Modelo não encontrado no caminho: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o modelo: {e}")

def load_video(video_path: Path) -> cv2.VideoCapture:
    """
    Carrega o vídeo a partir do caminho especificado.

    Args:
        video_path (Path): Caminho para o arquivo de vídeo.

    Returns:
        cv2.VideoCapture: Objeto de captura de vídeo.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado no caminho: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    return cap

def extract_detections(results) -> Optional[Tuple[np.ndarray, list, list, list]]:
    """
    Extrai as detecções do resultado do modelo YOLO.

    Args:
        results: Resultado da detecção do modelo YOLO.

    Returns:
        Optional[Tuple[np.ndarray, list, list, list]]: Tupla com as detecções ou None se não houver detecções.
    """
    boxes_obj = results.boxes
    if boxes_obj is None or boxes_obj.id is None:
        return None

    boxes = boxes_obj.xyxy.cpu().numpy()
    ids = boxes_obj.id.int().cpu().tolist()
    classes = boxes_obj.cls.int().cpu().tolist()
    confs = boxes_obj.conf.cpu().tolist()

    return boxes, ids, classes, confs

def cross_line(prev_center: Optional[Center], current_y: int, line_y: int) -> bool:
    """
    Verifica se o centro do veículo cruzou a linha de contagem.

    Args:
        prev_center (Optional[Center]): Centro anterior do veículo.
        current_y (int): Coordenada y atual do centro do veículo.
        line_y (int): Coordenada y da linha de contagem.

    Returns:
        bool: True se o veículo cruzou a linha, False caso contrário.
    """
    if prev_center is None:
        return False

    prev_y = prev_center[1]

    if DIRECAO_CONTAGEM == "descendo":
        return prev_y < line_y <= current_y
    elif DIRECAO_CONTAGEM == "subindo":
        return prev_y > line_y >= current_y
    elif DIRECAO_CONTAGEM == "ambos":
        return (prev_y < line_y <= current_y) or (prev_y > line_y >= current_y)

    return False

def draw_annotations(frame: np.ndarray, box: np.ndarray, center: Center, obj_id: int, cls: int, conf: float) -> None:
    """
    Desenha as anotações no frame: retângulo delimitador, ID do objeto,
    classe e confiança.

    Args:
        frame (np.ndarray): Frame atual.
        box (np.ndarray): Caixa delimitadora do objeto.
        center (Center): Centro do objeto.
        obj_id (int): ID do objeto.
        cls (int): Classe do objeto.
        conf (float): Confiança da detecção.
    """
    x1, y1, x2, y2 = map(int, box)
    cx, cy = center
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    label = f"ID: {obj_id} | Class: {cls} | Conf: {conf:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def process_frame(frame: np.ndarray, results, line_y: int, counted_ids: Set[int], last_centers: Dict[int, Center]) -> int:
    """
    Processa um frame: atualiza tracking, verifica cruzamentos de linha
    e desenha as anotações. Retorna quantos veículos novos foram
    contados nesse frame.
    """

    detections = extract_detections(results)
    new_count = 0
    if detections is None:
        return new_count

    boxes, ids, classes, confs = detections
    for box, obj_id, cls, conf in zip(boxes, ids, classes, confs):
        if cls not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box)
        center: Center = ((x1 + x2) // 2, (y1 + y2) // 2)

        prev_center = last_centers.get(obj_id)

        if cross_line(prev_center, center[1], line_y) and obj_id not in counted_ids:
            counted_ids.add(obj_id)
            new_count += 1
            print(f"Veículo detectado! ID: {obj_id} | Classe: {cls}")

        last_centers[obj_id] = center
        draw_annotations(frame, box, center, obj_id, cls, conf)
    return new_count

def draw_overlay(frame: np.ndarray, width: int, line_y: int, count: int) -> None:
    """ Desenha a linha de contagem e o contador no frame. """
    cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)
    cv2.putText(frame, f"Contagem: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# ------------------------------ Função principal ------------------------------ #

def main():
    model = load_model(MODEL_PATH)

    cap = load_video(VIDEO_PATH)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * LINE_Y_RATIO)

    counted_ids: Set[int] = set()
    last_centers: Dict[int, Center] = {}
    count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Fim do vídeo ou erro na leitura do frame.")
                break

            
            try:
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONFIDENCE, verbose=False)
            except Exception as exc:
                print(f"Erro durante a detecção: {exc}")
                break

            if not results or len(results) == 0:
                print("Nenhum resultado de detecção.")
                continue

            count += process_frame(frame, results[0], line_y, counted_ids, last_centers)
            draw_overlay(frame, width, line_y, count)

            cv2.imshow("YOLOv8 Tracking", frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

    except Exception as e:
        print(f"Erro durante a execução: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal final de veículos contados: {count}")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"Erro: {exc}")
    except RuntimeError as exc:
        print(f"Erro: {exc}")
    except Exception as exc:
        print(f"Erro inesperado: {exc}")