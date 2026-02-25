import numpy as np

from PANDORA.PRDA.lib.ImageRecorder import ImageRecorder


def generate_bfov_mask_cpu(bfov_params, erp_w=1920, erp_h=960):
    """
    生成单个BFOV的掩码（CPU版本）
    
    参数:
    bfov_params: BFOV参数 [longitude, latitude, fov_x, fov_y]（弧度制）
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    
    返回:
    mask: [H, W]的numpy数组，H=erp_h, W=erp_w
    """
    lon, lat, fov_x, fov_y = bfov_params
    
    if fov_x > np.pi or fov_y > np.pi:
        return None
    
    fov_x_deg = np.degrees(fov_x)
    fov_y_deg = np.degrees(fov_y)
    
    cpu_recorder = ImageRecorder(erp_w, erp_h, view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, long_side=erp_w)
    
    Px, Py = cpu_recorder._sample_points(lon, lat, border_only=False)
    
    Px = Px.astype(np.int32)
    Py = Py.astype(np.int32)
    
    valid_mask = (Px >= 0) & (Px < erp_w) & (Py >= 0) & (Py < erp_h)
    valid_Px = Px[valid_mask]
    valid_Py = Py[valid_mask]
    
    if len(valid_Px) == 0:
        return None
    
    mask = np.zeros((erp_h, erp_w), dtype=np.uint8)
    mask[valid_Py, valid_Px] = 1
    
    return mask


def compute_bounding_box(mask):
    """
    计算掩码的外接矩形包围框
    
    参数:
    mask: [H, W]的numpy数组
    
    返回:
    bbox: [x, y, width, height] 包围框参数
          x: 左上角x坐标
          y: 左上角y坐标
          width: 宽度
          height: 高度
    """
    if mask is None or mask.size == 0 or len(mask.shape) == 0:
        return None
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    y_min = np.where(rows)[0][0]
    y_max = np.where(rows)[0][-1]
    
    x_indices = np.where(cols)[0]
    x_min = x_indices[0]
    x_max = x_indices[-1]
    
    erp_w = mask.shape[1]
    x_diff = x_max - x_min
    
    if x_diff > erp_w * 0.6:
        left_mask = mask.copy()
        left_mask[:, erp_w//2:] = 0
        
        right_mask = mask.copy()
        right_mask[:, :erp_w//2] = 0
        
        left_rows = np.any(left_mask, axis=1)
        left_cols = np.any(left_mask, axis=0)
        
        right_rows = np.any(right_mask, axis=1)
        right_cols = np.any(right_mask, axis=0)
        
        left_area = 0
        right_area = 0
        
        if np.any(left_rows) and np.any(left_cols):
            left_x_min = np.where(left_cols)[0][0]
            left_x_max = np.where(left_cols)[0][-1]
            left_y_min = np.where(left_rows)[0][0]
            left_y_max = np.where(left_rows)[0][-1]
            left_area = (left_x_max - left_x_min + 1) * (left_y_max - left_y_min + 1)
        
        if np.any(right_rows) and np.any(right_cols):
            right_x_min = np.where(right_cols)[0][0]
            right_x_max = np.where(right_cols)[0][-1]
            right_y_min = np.where(right_rows)[0][0]
            right_y_max = np.where(right_rows)[0][-1]
            right_area = (right_x_max - right_x_min + 1) * (right_y_max - right_y_min + 1)
        
        if left_area > right_area:
            x_min = left_x_min
            x_max = left_x_max
        else:
            x_min = right_x_min
            x_max = right_x_max
    
    x = x_min
    y = y_min
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    
    return [x, y, width, height]


def bfov_to_bbox(bfov_params, erp_w=1920, erp_h=960):
    """
    将BFOV参数转换为包围框（核心函数）
    
    参数:
    bfov_params: BFOV参数 [longitude, latitude, fov_x, fov_y]（弧度制）
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    
    返回:
    bbox: [x, y, width, height] 包围框参数
          x: 左上角x坐标
          y: 左上角y坐标
          width: 宽度
          height: 高度
    """
    mask = generate_bfov_mask_cpu(bfov_params, erp_w, erp_h)
    bbox = compute_bounding_box(mask)
    return bbox


def batch_bfov_to_bbox(bfov_list, erp_w=1920, erp_h=960):
    """
    批量将BFOV参数转换为包围框
    
    参数:
    bfov_list: BFOV参数列表，每个元素是 [longitude, latitude, fov_x, fov_y]（弧度制）
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    
    返回:
    bbox_list: 包围框列表，每个元素是 [x, y, width, height]
    """
    bbox_list = []
    for bfov_params in bfov_list:
        bbox = bfov_to_bbox(bfov_params, erp_w, erp_h)
        bbox_list.append(bbox)
    return bbox_list
