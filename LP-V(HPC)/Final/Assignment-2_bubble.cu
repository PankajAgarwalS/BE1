#include <stdio.h>

// Kernel function to swap two elements
__device__ void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// CUDA kernel for bubble sort
__global__ void bubbleSort(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    swap(&arr[j], &arr[j + 1]);
                }
            }
        }
    }
}

int main() {
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    int *arr, *d_arr;
    size_t size = n * sizeof(int);

    // Allocate memory for host and device arrays
    arr = (int *)malloc(size);
    cudaMalloc(&d_arr, size);

    // Input array elements
    printf("Enter %d elements:\n", n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    // Copy array from host to device
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    // Call CUDA kernel
    bubbleSort<<<grid_size, block_size>>>(d_arr, n);

    // Copy array from device to host
    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);

    // Display sorted array
    printf("Sorted array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    // Free memory
    free(arr);
    cudaFree(d_arr);

    return 0;
}
