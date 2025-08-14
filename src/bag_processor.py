#python src/bag_processor.py --input data/raw_bags/1018142132.bag --output data/generated

import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
import os

def process_bag(bag_path: str, output_dir: str):
    output_dir = Path(output_dir)
    (output_dir / "depth").mkdir(parents=True, exist_ok=True)
    (output_dir / "generatedIR").mkdir(parents=True, exist_ok=True)
    (output_dir / "generatedRGB").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth_aligned").mkdir(parents=True, exist_ok=True)  # NOVO: pasta para depth alinhado

    # Inicializa pipeline e configura para ler do .bag
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)
    config.enable_all_streams()
    pipeline.start(config)

    # Obtém perfis para salvar intrínsecos
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color)
    depth_stream = profile.get_stream(rs.stream.depth)
    try:
        ir_stream = profile.get_stream(rs.stream.infrared)
    except Exception:
        ir_stream = None

    # Salva intrínsecos
    color_intr = color_stream.as_video_stream_profile().get_intrinsics()
    K = np.array([color_intr.fx, 0, color_intr.ppx,
              0, color_intr.fy, color_intr.ppy,
              0, 0, 1], dtype=np.float32)
    np.save(output_dir / "camera_matrix_color.npy", K)
    depth_intr = depth_stream.as_video_stream_profile().get_intrinsics()
    K_depth = np.array([
        depth_intr.fx, 0, depth_intr.ppx,
        0, depth_intr.fy, depth_intr.ppy,
        0, 0, 1
    ], dtype=np.float32)
    np.save(output_dir / "camera_matrix_depth.npy", K_depth)
    if ir_stream:
        ir_intr = ir_stream.as_video_stream_profile().get_intrinsics()
        K_ir = np.array([ir_intr.fx, 0, ir_intr.ppx,
                         0, ir_intr.fy, ir_intr.ppy,
                         0, 0, 1], dtype=np.float32)
        np.save(output_dir / "camera_matrix_ir.npy", K_ir)

    # CLAHE para IR
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    depth_idx = 0
    ir_idx = 0
    rgb_idx = 0

    # NOVO: alinhador de depth para color
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            # Alinha todos os frames ao color
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Depth (original)
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_img = np.asanyarray(depth_frame.get_data())
                depth_img = cv2.medianBlur(depth_img, 5)
                np.save(output_dir / "depth" / f"{depth_idx:06d}.npy", depth_img)

            # Depth (alinhado ao RGB)
            if aligned_depth_frame:
                depth_aligned_img = np.asanyarray(aligned_depth_frame.get_data())
                depth_aligned_img = cv2.medianBlur(depth_aligned_img, 5)
                np.save(output_dir / "depth_aligned" / f"{depth_idx:06d}.npy", depth_aligned_img)

            depth_idx += 1

            # IR
            ir_frame = frames.get_infrared_frame()
            if ir_frame:
                ir_img = np.asanyarray(ir_frame.get_data())
                ir_img = cv2.equalizeHist(ir_img)
                enhanced_ir = clahe.apply(ir_img)
                cv2.imwrite(str(output_dir / "generatedIR" / f"{ir_idx:06d}.png"), enhanced_ir)
                ir_idx += 1

            # RGB (alinhado)
            if color_frame:
                color_img = np.asanyarray(color_frame.get_data())  # Já está em BGR
                cv2.imwrite(str(output_dir / "generatedRGB" / f"{rgb_idx:06d}.png"), color_img)
                rgb_idx += 1

    except RuntimeError:
        # Fim do arquivo .bag
        pass
    finally:
        pipeline.stop()

def export_rgb_video(output_dir: str, video_filename: str = "rgb_video.mp4", fps: int = 30):
    rgb_dir = Path(output_dir) / "generatedRGB"
    img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    if not img_files:
        print("Nenhum frame RGB encontrado.")
        return
    first_frame = cv2.imread(str(rgb_dir / img_files[0]))
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(str(Path(output_dir) / video_filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for img_file in img_files:
        frame = cv2.imread(str(rgb_dir / img_file))
        out.write(frame)
    out.release()
    print(f"Vídeo salvo em {output_dir}/{video_filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Processa arquivo .bag para extrair frames Depth (.npy), IR (generatedIR) e RGB (generatedRGB) usando pyrealsense2.")
    parser.add_argument("--input", required=True, help="Caminho para o arquivo .bag")
    parser.add_argument("--output", required=True, help="Diretório de saída dos frames processados")
    args = parser.parse_args()

    process_bag(args.input, args.output)
    export_rgb_video(args.output)