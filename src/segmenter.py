#python src/segmenter.py --input_dir data/generated --output_dir results --model_path models/yolov8n-seg.pt

import os
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from ultralytics import YOLO
from typing import Tuple, List

class SheepSegmenter:
    def __init__(self, model_path: str):
        """Inicializa o segmentador YOLOv8"""
        self.model = YOLO(model_path)
        self.class_id = 21  # Ajuste conforme sua classe de interesse

    def enhance_ir(self, image: np.ndarray) -> np.ndarray:
        """Melhora o contraste de imagens IR"""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(cv2.equalizeHist(image))

    def segment_frame(self, image: np.ndarray) -> np.ndarray:
        """Executa segmentação 2D com YOLOv8, aceitando IR (1 canal) ou RGB (3 canais)"""
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            # IR: aplica realce e converte para pseudo-RGB
            enhanced = self.enhance_ir(image)
            pseudo_rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)
            input_img = pseudo_rgb
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # RGB: usa direto
            input_img = image
        else:
            raise ValueError("Formato de imagem não suportado para segmentação.")

        results = self.model(input_img)
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for result in results:
            for i, cls in enumerate(result.boxes.cls):
                if int(cls) == self.class_id:
                    mask = result.masks[i].data.cpu().numpy().squeeze()
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    combined_mask = np.logical_or(combined_mask, mask_resized)
        return (combined_mask * 255).astype(np.uint8)

class PointCloudProcessor:
    def __init__(self, camera_matrix: np.ndarray):
        """Inicializa o processador de nuvem de pontos"""
        self.K = camera_matrix

    def depth_to_3d(self, depth_frame: np.ndarray) -> np.ndarray:
        """Converte frame de profundidade para coordenadas 3D"""
        fx, fy = self.K[0], self.K[4]
        cx, cy = self.K[2], self.K[5]
        height, width = depth_frame.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_frame.astype(np.float32) / 1000.0  # mm -> metros
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack((x, y, z), axis=-1)  # shape: (H, W, 3)

    def mask_pointcloud(self, points_3d: np.ndarray, mask_2d: np.ndarray) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
        """Aplica máscara 2D à nuvem de pontos 3D, removendo pontos inválidos"""
        # Garante que a máscara tem o mesmo shape da imagem
        if mask_2d.shape != points_3d.shape[:2]:
            mask_2d = cv2.resize(mask_2d, (points_3d.shape[1], points_3d.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_2d > 127
        sheep_points = points_3d[mask_bool]
        # Remove pontos com profundidade inválida (z <= 0)
        sheep_points = sheep_points[sheep_points[:, 2] > 0]
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
    """Pipeline completo de segmentação"""
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pointclouds"), exist_ok=True)

    # Carregar parâmetros da câmera (profundidade)
    camera_matrix = np.load(os.path.join(input_dir, "camera_matrix_depth.npy"))
    if camera_matrix.size == 9:
        K = camera_matrix
    elif camera_matrix.shape == (3, 3):
        K = camera_matrix.flatten()
    else:
        raise ValueError("camera_matrix_depth.npy deve ter 9 elementos (fx,0,ppx,0,fy,ppy,0,0,1)")

    segmenter = SheepSegmenter(model_path)
    pc_processor = PointCloudProcessor(K)

    # Detecta se vai usar IR ou RGB
    ir_dir = os.path.join(input_dir, "generatedIR")
    rgb_dir = os.path.join(input_dir, "generatedRGB")
    depth_dir = os.path.join(input_dir, "depth")

    if os.path.isdir(ir_dir) and len(os.listdir(ir_dir)) > 0:
        img_dir = ir_dir
        img_mode = "IR"
        img_files = sorted([f for f in os.listdir(ir_dir) if f.endswith('.png')])
        img_read_flag = cv2.IMREAD_GRAYSCALE
    elif os.path.isdir(rgb_dir) and len(os.listdir(rgb_dir)) > 0:
        img_dir = rgb_dir
        img_mode = "RGB"
        img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        img_read_flag = cv2.IMREAD_COLOR
    else:
        raise RuntimeError("Nenhuma imagem IR ou RGB encontrada no diretório de entrada.")

    for img_file in tqdm(img_files, desc=f"Processando frames ({img_mode})"):
        frame_id = os.path.splitext(img_file)[0]
        img = cv2.imread(os.path.join(img_dir, img_file), img_read_flag)
        depth_path = os.path.join(depth_dir, f"{frame_id}.npy")
        if not os.path.exists(depth_path):
            continue
        depth_frame = np.load(depth_path)

        # Segmentação 2D
        mask = segmenter.segment_frame(img)

        # Processamento 3D
        points_3d = pc_processor.depth_to_3d(depth_frame)
        sheep_points, sheep_pcd = pc_processor.mask_pointcloud(points_3d, mask)

        # Pós-processamento 3D
        if len(sheep_pcd.points) > 0:
            sheep_pcd = sheep_pcd.voxel_down_sample(voxel_size=0.01)
            sheep_pcd, _ = sheep_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Salvar resultados
        if save_results:
            cv2.imwrite(os.path.join(output_dir, "masks", f"{frame_id}_mask.png"), mask)
            if len(sheep_pcd.points) > 0:
                o3d.io.write_point_cloud(
                    os.path.join(output_dir, "pointclouds", f"{frame_id}_sheep.pcd"),
                    sheep_pcd
                )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Diretório com dados processados (deve conter 'generatedIR' ou 'generatedRGB' e 'depth')")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Diretório para salvar resultados")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Caminho para o modelo YOLOv8 (.pt)")
    parser.add_argument("--visualize", action="store_true",
                       help="Mostrar visualizações interativas")
    args = parser.parse_args()

    process_frames(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        visualize=args.visualize
    )