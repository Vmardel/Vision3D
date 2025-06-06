import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import open3d as o3d

# ---------------------------
# Block Matching Class (BM)
# ---------------------------

class BM:
    def __init__(self, kernel_size, max_disparity, subpixel_interpolation):
        self.kernel_size = kernel_size
        self.max_disparity = max_disparity
        self.kernel_half = kernel_size // 2
        self.offset_adjust = 255 / max_disparity
        self.subpixel_interpolation = subpixel_interpolation

    def _get_window(self, y, x, img, offset=0):
        y_start = y - self.kernel_half
        y_end = y + self.kernel_half
        x_start = x - self.kernel_half - offset + 1
        x_end = x + self.kernel_half - offset + 1
        return img[y_start:y_end, x_start:x_end]

    def _compute_subpixel_offset(self, best_offset, errors):
        errors = np.array(errors, dtype=np.float64)
        if (
            0 < best_offset < self.max_disparity - 1 and
            np.isfinite(errors[best_offset]) and
            np.isfinite(errors[best_offset - 1]) and
            np.isfinite(errors[best_offset + 1])
        ):
            denom = errors[best_offset - 1] + errors[best_offset + 1] - 2 * errors[best_offset]
            if denom != 0:
                numerator = errors[best_offset - 1] - errors[best_offset + 1]
                subpixel = numerator / (2 * denom)
                if -1.0 <= subpixel <= 1.0:
                    return subpixel
        return 0.0

    def compute(self, left, right):
        h, w = left.shape
        disp_map = np.zeros_like(left, dtype=np.float32)

        for y in range(self.kernel_half, h - self.kernel_half):
            for x in range(self.max_disparity, w - self.kernel_half):
                best_offset = 0
                min_error = float("inf")
                errors = []

                for offset in range(self.max_disparity):
                    W_left = self._get_window(y, x, left)
                    W_right = self._get_window(y, x, right, offset)

                    if W_left.shape != W_right.shape:
                        errors.append(np.inf)
                        continue

                    error = np.sum((W_left - W_right) ** 2)
                    errors.append(error)

                    if error < min_error:
                        min_error = error
                        best_offset = offset

                if self.subpixel_interpolation:
                    best_offset += self._compute_subpixel_offset(best_offset, errors)

                disp_map[y, x] = best_offset * self.offset_adjust

        return disp_map

# ---------------------------
# Utilities
# ---------------------------

def load_images(root):
    left = Image.open(f"{root}/left.png").convert("RGB")
    right = Image.open(f"{root}/right.png").convert("RGB")
    gt = Image.open(f"{root}/gt.png").convert("L")
    return left, right, gt

def compute_metrics(pred, gt, threshold=3.0):
    gt = np.array(gt)
    mask = gt > 0
    pred_valid = pred[mask].astype(np.float32)
    gt_valid = gt[mask].astype(np.float32)

    abs_error = np.abs(pred_valid - gt_valid)
    epe = np.mean(abs_error)
    bad_pixels = np.sum(abs_error > threshold) / len(gt_valid) * 100
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    mae = np.mean(abs_error)

    return {
        "EPE": round(float(epe), 2),
        "Bad Pixel %": round(float(bad_pixels), 2),
        "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 2)
    }

def save_point_cloud(filename, disparity, colors):
    Q = np.array([[1, 0, 0, -disparity.shape[1] / 2],
                  [0, -1, 0, disparity.shape[0] / 2],
                  [0, 0, 0, -0.8 * disparity.shape[1]],
                  [0, 0, 1 / 0.05, 0]])

    points_3d = cv.reprojectImageTo3D(disparity, Q)
    mask = disparity > 0
    points = points_3d[mask]
    colors = colors[mask]
    points = np.hstack([points, colors])
    
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%f %f %f %d %d %d")

def create_pointcloud(disparity, colors):
    Q = np.array([[1, 0, 0, -disparity.shape[1] / 2],
                  [0, -1, 0, disparity.shape[0] / 2],
                  [0, 0, 0, -0.8 * disparity.shape[1]],
                  [0, 0, 1 / 0.05, 0]])

    points_3d = cv.reprojectImageTo3D(disparity, Q)
    mask = disparity > 0

    points = points_3d[mask]
    colors = colors[mask].astype(np.float64) / 255.0  # Normalizar colores a [0,1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

# ---------------------------
# Main Execution
# ---------------------------

def main():
    root = "Imagenes/Teddy"  # Cambia la ruta a donde estén tus imágenes
    left_img, right_img, gt_img = load_images(root)

    left_gray = np.array(left_img.convert("L"))
    right_gray = np.array(right_img.convert("L"))
    gt = np.array(gt_img)

    bm = BM(kernel_size=5, max_disparity=64, subpixel_interpolation=True)
    disparity = bm.compute(left_gray, right_gray)

    metrics = compute_metrics(disparity, gt)
    print("Métricas de disparidad:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    plt.imshow(disparity, cmap="plasma")
    plt.title("Mapa de Disparidad")
    plt.colorbar()
    plt.show()

    color_array = np.array(left_img)
    save_point_cloud("pointcloud.ply", disparity, color_array)
    print("Nube de puntos guardada como pointcloud.ply")

    create_pointcloud(disparity, color_array)  # Visualiza la nube con Open3D

if __name__ == "__main__":
    main()
