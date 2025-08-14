import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("results/pointclouds/000015_sheep.pcd")

# 1. Bounding box do espaço inteiro (AABB de todos os pontos)
aabb_full = pcd.get_axis_aligned_bounding_box()
aabb_full.color = (1, 0, 0)  # vermelho

# 2. Bounding box do animal (maior cluster acima do chão)
points = np.asarray(pcd.points)
mask_above_ground = points[:, 2] > 0  # considera só pontos acima do chão
pcd_above = pcd.select_by_index(np.where(mask_above_ground)[0])

labels = np.array(pcd_above.cluster_dbscan(eps=0.05, min_points=10, print_progress=True))
if labels.max() >= 0:
    largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))
    idx_main = np.where(labels == largest_cluster)[0]
    pcd_animal = pcd_above.select_by_index(idx_main)
    aabb_animal = pcd_animal.get_axis_aligned_bounding_box()
    aabb_animal.color = (0, 1, 0)  # verde
    geometries = [pcd, aabb_full, aabb_animal]
else:
    geometries = [pcd, aabb_full]

o3d.visualization.draw_geometries(geometries)