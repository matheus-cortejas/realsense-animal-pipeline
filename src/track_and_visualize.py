#python src/track_and_visualize.py --video data/generated/rgb_video.mp4 --masks results/masks --output results/tracked_video.mp4

import cv2
import numpy as np
import os
from sort import Sort

def get_bboxes_from_masks(masks_dir, img_shape, failed_txt_path=None):
    """Lê as máscaras e retorna bounding boxes para cada frame. Salva nomes das máscaras que falharam."""
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('_mask.png')])
    bboxes_per_frame = []
    failed_masks = []
    for mask_file in mask_files:
        mask = cv2.imread(os.path.join(masks_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            bboxes_per_frame.append([])
            failed_masks.append(mask_file)
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 100:  # ignora ruído
                bboxes.append([x, y, x + w, y + h, 1.0])  # formato SORT: [x1, y1, x2, y2, score]
        if not bboxes:
            failed_masks.append(mask_file)
        bboxes_per_frame.append(bboxes)
    if failed_txt_path and failed_masks:
        with open(failed_txt_path, "w") as f:
            for name in failed_masks:
                f.write(name + "\n")
    return mask_files, bboxes_per_frame

def track_and_draw(video_path, masks_dir, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    failed_txt_path = os.path.join(os.path.dirname(output_path), "failed_masks.txt")
    mask_files, bboxes_per_frame = get_bboxes_from_masks(masks_dir, (height, width), failed_txt_path)
    tracker = Sort()  # Inicializa SORT

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(bboxes_per_frame):
            break
        bboxes = np.array(bboxes_per_frame[frame_idx])
        if bboxes.shape[0] == 0:
            tracker.update()
        else:
            tracks = tracker.update(bboxes)
            for d in tracks:
                x1, y1, x2, y2, track_id = map(int, d)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f'ID {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Vídeo com tracking salvo em {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Caminho para o vídeo RGB (.mp4)")
    parser.add_argument("--masks", required=True, help="Diretório das máscaras segmentadas")
    parser.add_argument("--output", required=True, help="Caminho do vídeo de saída com tracking")
    args = parser.parse_args()

    track_and_draw(args.video, args.masks, args.output)