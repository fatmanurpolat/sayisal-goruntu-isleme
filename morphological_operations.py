import numpy as np

def erosion(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return morphological_operation(image, kernel, operation='erosion')

def dilation(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return morphological_operation(image, kernel, operation='dilation')

def opening(image, kernel_size=5):
    eroded = erosion(image, kernel_size)
    return dilation(eroded, kernel_size)

def closing(image, kernel_size=5):
    dilated = dilation(image, kernel_size)
    return erosion(dilated, kernel_size)

def morphological_operation(image, kernel, operation):
    pad_height, pad_width = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
    output = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                region = padded_image[y:y + kernel.shape[0], x:x + kernel.shape[1], c]
                if operation == 'erosion':
                    output[y, x, c] = np.min(region[kernel == 1])
                elif operation == 'dilation':
                    output[y, x, c] = np.max(region[kernel == 1])
    return output
