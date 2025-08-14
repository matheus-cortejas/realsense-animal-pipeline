#python src/segmenter.py --input_dir data/generated --output_dir results --model_path models/yolov8n-seg.pt

from tqdm import tqdm
from ultralytics import YOLO
from typing import Tuple, List
import numpy as np
import cv2
import os

class SheepSegmenter:
    def __init__(self, model_path: str):
        """Inicializa o segmentador YOLOv8"""
        self.model = YOLO(model_path)
        self.class_id = 21  # Ajuste conforme sua classe de interesse

    def enhance_ir(self, image: np.ndarray) -> np.ndarray:
        """Melhora o contraste de imagens IR"""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def segment_frame(self, image: np.ndarray) -> np.ndarray:
        """Segmenta o animal na imagem e retorna a máscara binária"""
        results = self.model(image)
        result = results[0]
        if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for i, cls in enumerate(result.boxes.cls):
            if int(cls) == self.class_id:
                pred_mask = result.masks.data[i].cpu().numpy().astype(np.uint8) * 255
                pred_mask = cv2.resize(pred_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = np.maximum(mask, pred_mask)
        return mask

class PointCloudProcessor:
    def __init__(self, camera_matrix: np.ndarray):
        self.K = camera_matrix

    def depth_to_3d(self, depth_frame: np.ndarray) -> np.ndarray:
        fx, fy = self.K[0], self.K[4]
        cx, cy = self.K[2], self.K[5]
        height, width = depth_frame.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_frame.astype(np.float32) / 1000.0  # mm -> metros
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack((x, y, z), axis=-1)  # shape: (H, W, 3)

    def mask_pointcloud(self, points_3d: np.ndarray, mask_2d: np.ndarray):
        if mask_2d.shape != points_3d.shape[:2]:
            mask_2d = cv2.resize(mask_2d, (points_3d.shape[1], points_3d.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_2d > 127
        sheep_points = points_3d[mask_bool]
        sheep_points = sheep_points[sheep_points[:, 2] > 0]
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        if sheep_points.shape[0] > 0:
            pcd.points = o3d.utility.Vector3dVector(sheep_points)
        return sheep_points, pcd

def process_frames(
    input_dir: str,
    output_dir: str,
    model_path: str,
    visualize: bool = False,
    save_results: bool = True
):
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pointclouds"), exist_ok=True)

    # Carregar parâmetros da câmera (profundidade)
    camera_matrix = np.load(os.path.join(input_dir, "camera_matrix_color.npy"))
    if camera_matrix.size == 9:
        K = camera_matrix
    elif camera_matrix.shape == (3, 3):
        K = camera_matrix.flatten()
    else:
        raise ValueError("camera_matrix_color.npy deve ter 9 elementos (fx,0,ppx,0,fy,ppy,0,0,1)")

    segmenter = SheepSegmenter(model_path)
    pc_processor = PointCloudProcessor(K)

    rgb_dir = os.path.join(input_dir, "generatedRGB")
    depth_dir = os.path.join(input_dir, "depth_aligned")  # Use o depth alinhado!
    img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])

    for img_file in tqdm(img_files, desc="Processando frames (RGB + Depth Alinhado)"):
        frame_id = os.path.splitext(img_file)[0]
        img = cv2.imread(os.path.join(rgb_dir, img_file), cv2.IMREAD_COLOR)
        depth_path = os.path.join(depth_dir, f"{frame_id}.npy")
        if not os.path.exists(depth_path):
            continue
        depth_frame = np.load(depth_path)

        # Segmentação 2D
        mask = segmenter.segment_frame(img)

        # Garante alinhamento de shape
        if mask.shape != depth_frame.shape:
            mask = cv2.resize(mask, (depth_frame.shape[1], depth_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Processamento 3D
        points_3d = pc_processor.depth_to_3d(depth_frame)
        sheep_points, sheep_pcd = pc_processor.mask_pointcloud(points_3d, mask)

        # Pós-processamento 3D
        if sheep_pcd.has_points():
            sheep_pcd = sheep_pcd.voxel_down_sample(voxel_size=0.01)
            sheep_pcd, _ = sheep_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Salvar resultados
        if save_results:
            cv2.imwrite(os.path.join(output_dir, "masks", f"{frame_id}_mask.png"), mask)
            if sheep_pcd.has_points():
                import open3d as o3d
                o3d.io.write_point_cloud(
                    os.path.join(output_dir, "pointclouds", f"{frame_id}_sheep.pcd"),
                    sheep_pcd
                )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Diretório de entrada dos frames")
    parser.add_argument("--output_dir", required=True, help="Diretório de saída dos resultados")
    parser.add_argument("--model_path", required=True, help="Caminho do modelo YOLOv8")
    args = parser.parse_args()
    process_frames(args.input_dir, args.output_dir, args.model_path)