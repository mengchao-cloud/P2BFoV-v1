#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

FIXED_FOV_PARAMS = [
    (0.2612193871, 0.1306096936),
    (0.1847100000, 0.1847100000),
    (0.1306096936, 0.2612193871),
    (0.6791704065, 0.3395852032),
    (0.4802460000, 0.4802460000),
    (0.3395852032, 0.6791704065),
    (1.2016091807, 0.6008045903),
    (0.8496660000, 0.8496660000),
    (0.6008045903, 1.2016091807),
]

PERSON_THETA = -0.0801760625134895
PERSON_PHI = -0.16853335589570229
CHAIR_THETA = 0.8492117641734908
CHAIR_PHI = -1.0750137361502572

FIXED_BFOV_ANNOTATIONS = []
for fov_x, fov_y in FIXED_FOV_PARAMS:
    FIXED_BFOV_ANNOTATIONS.append([34, PERSON_THETA, PERSON_PHI, fov_x, fov_y])
for fov_x, fov_y in FIXED_FOV_PARAMS:
    FIXED_BFOV_ANNOTATIONS.append([25, CHAIR_THETA, CHAIR_PHI, fov_x, fov_y])

ERP_WIDTH = 1920
ERP_HEIGHT = 960
IMAGE_PATH = './7PhJB.jpg'
GRID_SIZE = 7
OUTPUT_DIR = "./bfov_shaken_output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def bfov_roi_to_7x7_erp(bfov_params, grid_size=7):
    lon, lat, fov_u, fov_v = bfov_params[1:5]
    PI = np.pi
    
    phi0 = lon
    theta0 = PI/2 - lat
    
    c_x = np.sin(theta0) * np.cos(phi0)
    c_y = np.sin(theta0) * np.sin(phi0)
    c_z = np.cos(theta0)
    c = np.array([c_x, c_y, c_z])
    c = c / np.linalg.norm(c)
    
    sin_phi = np.sin(phi0)
    cos_phi = np.cos(phi0)
    sin_theta = np.sin(theta0)
    cos_theta = np.cos(theta0)
    
    lon_dir = np.array([-sin_phi, cos_phi, 0])
    lon_dir = lon_dir / np.linalg.norm(lon_dir)
    
    lat_dir = np.array([-cos_theta*cos_phi, -cos_theta*sin_phi, sin_theta])
    lat_dir = lat_dir / np.linalg.norm(lat_dir)
    
    dot_product = np.dot(lon_dir, lat_dir)
    if abs(dot_product) > 1e-6:
        lon_dir_ortho = lon_dir - np.dot(lon_dir, c) * c
        lon_dir_ortho = lon_dir_ortho / np.linalg.norm(lon_dir_ortho)
        lat_dir_ortho = np.cross(c, lon_dir_ortho)
        lat_dir_ortho = lat_dir_ortho / np.linalg.norm(lat_dir_ortho)
        lon_dir, lat_dir = lon_dir_ortho, lat_dir_ortho
    
    max_angle = PI/2 - 1e-4
    fov_u_half = np.clip(fov_u / 2.0, -max_angle, max_angle)
    fov_v_half = np.clip(fov_v / 2.0, -max_angle, max_angle)
    
    U = np.tan(fov_u_half)
    V = np.tan(fov_v_half)
    
    delta_u = 2 * U / grid_size
    delta_v = 2 * V / grid_size
    
    j_grid, i_grid = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    
    j_plus_half = j_grid + 0.5
    vertical_idx = (grid_size - 1 - i_grid) + 0.5
    
    u_center = -U + j_plus_half * delta_u
    v_center = -V + vertical_idx * delta_v
    
    c_expand = np.tile(c, (grid_size, grid_size, 1))
    e_u_expand = np.tile(lon_dir, (grid_size, grid_size, 1))
    e_v_expand = np.tile(lat_dir, (grid_size, grid_size, 1))
    
    u_center_expand = np.expand_dims(u_center, axis=2)
    v_center_expand = np.expand_dims(v_center, axis=2)
    x_plane = c_expand + u_center_expand * e_u_expand + v_center_expand * e_v_expand
    
    norm_x_plane = np.linalg.norm(x_plane, axis=2, keepdims=True)
    x_sphere = x_plane / norm_x_plane
    
    x = x_sphere[..., 0]
    y = x_sphere[..., 1]
    z = x_sphere[..., 2]
    
    z_clamped = np.clip(z, -1.0 + 1e-6, 1.0 - 1e-6)
    
    theta = np.arccos(z_clamped)
    phi = np.arctan2(y, x)
    
    lat_deg = 90.0 - (theta * 180.0 / PI)
    lon_deg = phi * 180.0 / PI
    
    erp_x = (lon_deg + 180.0) / 360.0 * ERP_WIDTH
    erp_y = (90.0 - lat_deg) / 180.0 * ERP_HEIGHT
    
    erp_x = np.clip(erp_x, 0, ERP_WIDTH - 1)
    erp_y = np.clip(erp_y, 0, ERP_HEIGHT - 1)
    
    return erp_x, erp_y

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"错误：无法读取图像 {IMAGE_PATH}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (ERP_WIDTH, ERP_HEIGHT))
    
    fig, ax = plt.subplots(figsize=(ERP_WIDTH/100, ERP_HEIGHT/100), dpi=100)
    ax.imshow(img)
    ax.set_xlim(0, ERP_WIDTH)
    ax.set_ylim(ERP_HEIGHT, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    colors_person = plt.cm.Blues(np.linspace(0.6, 0.9, 6))
    colors_chair = [(1.0, 0.8, 0.0, 1.0), (1.0, 0.85, 0.0, 1.0), (1.0, 0.9, 0.0, 1.0), 
                    (1.0, 0.95, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0), (0.95, 1.0, 0.0, 1.0)]
    
    for i, bfov_params in enumerate(FIXED_BFOV_ANNOTATIONS):
        if i < 3:
            continue
        
        category_id = bfov_params[0]
        erp_x, erp_y = bfov_roi_to_7x7_erp(bfov_params, GRID_SIZE)
        
        if category_id == 34:
            color_idx = i - 3
            color = colors_person[color_idx]
        else:
            color_idx = i - 12
            color = colors_chair[color_idx]
        
        erp_x_flat = erp_x.flatten()
        erp_y_flat = erp_y.flatten()
        
        ax.scatter(erp_x_flat, erp_y_flat, c=[color], s=30, alpha=0.8, edgecolors='black', linewidths=0.5)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    output_file = os.path.join(OUTPUT_DIR, "7PhJB_7x7_sampling.png")
    plt.savefig(output_file, format='png', dpi=100)
    print(f"Saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    main()
