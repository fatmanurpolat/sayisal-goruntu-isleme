import numpy as np

def convolve2d(input_matrix, kernel):
    input_matrix = np.array(input_matrix)
    kernel = np.array(kernel)
    
    # Girdi matrisinin ve çekirdek matrisinin boyutları
    input_dim = input_matrix.shape
    kernel_dim = kernel.shape
    
    # Çıktı matrisinin boyutlarını hesapla
    output_dim = (input_dim[0] - kernel_dim[0] + 1, input_dim[1] - kernel_dim[1] + 1)
    output_matrix = np.zeros(output_dim)
    
    # Konvolüsyon işlemini gerçekleştir
    for i in range(output_dim[0]):
        for j in range(output_dim[1]):
            region = input_matrix[i:i+kernel_dim[0], j:j+kernel_dim[1]]
            output_matrix[i, j] = np.sum(region * kernel)
    
    return output_matrix

# Girdi matrisi
input_matrix = [
    [1, 2, 3, 0],
    [0, 1, 2, 3],
    [3, 0, 1, 2],
    [2, 3, 0, 1]
]

# Konvolüsyon çekirdeği (kernel)
kernel = [
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
]

# Çıktı matrisini hesapla
output_matrix = convolve2d(input_matrix, kernel)
print("Çıktı Matrisi:")
print(output_matrix)
