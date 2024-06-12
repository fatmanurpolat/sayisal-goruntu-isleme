import numpy as np

def region_growing(image, seed_point, threshold):
    rows, cols = image.shape[:2]
    segmented_image = np.zeros_like(image, np.uint8)
    seed_list = [seed_point]

    while len(seed_list) > 0:
        x, y = seed_list.pop(0)
        segmented_image[x, y] = 255  # Segmented bölgeyi beyaz yap

        # 8 bağlantılı komşular
        neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y), (x+1, y+1)]
        
        for nx, ny in neighbors:
            if 0 <= nx < rows and 0 <= ny < cols and segmented_image[nx, ny] == 0:
                if np.abs(int(image[nx, ny]) - int(image[x, y])) < threshold:
                    seed_list.append((nx, ny))
                    segmented_image[nx, ny] = 255

    return segmented_image
