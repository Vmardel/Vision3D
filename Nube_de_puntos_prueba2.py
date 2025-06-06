import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.ndimage import gaussian_filter

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

def bilateral_filter_grayscale(img, sigma_spatial=3, sigma_range=0.1):
    from scipy.ndimage import generic_filter

    def filter_func(patch):
        center = patch[len(patch) // 2]
        spatial_weights = np.exp(-0.5 * (np.arange(len(patch)) - len(patch) // 2) ** 2 / sigma_spatial ** 2)
        range_weights = np.exp(-0.5 * ((patch - center) ** 2) / (sigma_range ** 2))
        weights = spatial_weights * range_weights
        return np.sum(patch * weights) / np.sum(weights)

    filtered = generic_filter(img, filter_func, size=3, mode="reflect")
    return filtered

def save_disparity_image(disparity, filename="output/disparidad_mejorada.png"):
    disparity_filtered = bilateral_filter_grayscale(disparity, sigma_spatial=2, sigma_range=0.1)

    min_disp = np.min(disparity_filtered)
    max_disp = np.max(disparity_filtered)
    disparity_norm = (disparity_filtered - min_disp) / (max_disp - min_disp + 1e-6)

    # Sharpening sutil
    blurred = gaussian_filter(disparity_norm, sigma=0.3)
    sharpened = np.clip(disparity_norm + 0.5 * (disparity_norm - blurred), 0.2, 1)

    disp_img = Image.fromarray(np.uint8(sharpened * 255), mode="L")
    disp_img.save(filename)
    print(f"Disparidad guardada como: {filename}")

def create_pointcloud(disparity, colors, focal_length=1.0, baseline=0.05):
    h, w = disparity.shape
    mask = disparity > 0
    indices = np.indices((h, w), dtype=np.float32)
    x_coords = indices[1][mask]
    y_coords = indices[0][mask]
    disp = disparity[mask]

    Z = (focal_length * baseline) / disp
    X = (x_coords - w / 2) * Z / focal_length
    Y = (y_coords - h / 2) * Z / focal_length

    points = np.stack((X, Y, Z), axis=1)
    colors = colors[mask].astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def main():
    path_left = "/mnt/c/Users/vicen/Documents/GitHub/Vision3D/output/rect_hartley_izq.png"
    path_right = "/mnt/c/Users/vicen/Documents/GitHub/Vision3D/output/rect_hartley_der.png"

    left_img = Image.open(path_left).convert("RGB")
    right_img = Image.open(path_right).convert("RGB")

    left_gray = np.array(left_img.convert("L"))
    right_gray = np.array(right_img.convert("L"))

    bm = BM(kernel_size=5, max_disparity=64, subpixel_interpolation=True)
    disparity = bm.compute(left_gray, right_gray)

    save_disparity_image(disparity)
    create_pointcloud(disparity, np.array(left_img))

if __name__ == "__main__":
    main()