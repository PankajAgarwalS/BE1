#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000 // Number of nodes
#define BLOCK_SIZE 256

// Graph structure on device
__device__ int graph[N][N];
__device__ int visited[N];

// DFS kernel
__global__ void dfs(int start) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == start && !visited[tid]) {
        visited[tid] = 1;
        for (int i = 0; i < N; ++i) {
            if (graph[tid][i] && !visited[i]) {
                dfs<<<1, BLOCK_SIZE>>>(i);
            }
        }
    }
}

int main() {
    int graph_host[N][N]; // Graph structure on host
    int *visited_host; // Array to keep track of visited nodes on host
    int start_node = 0; // Starting node for DFS

    // Initialize graph randomly (example)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            graph_host[i][j] = rand() % 2; // Random binary graph
        }
    }

    // Allocate memory for visited array on host
    visited_host = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) {
        visited_host[i] = 0;
    }

    // Copy graph from host to device
    cudaMemcpyToSymbol(graph, graph_host, N * N * sizeof(int));

    // Copy visited array from host to device
    cudaMemcpyToSymbol(visited, visited_host, N * sizeof(int));

    // Measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch DFS kernel
    dfs<<<1, BLOCK_SIZE>>>(start_node);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print execution time
    printf("Execution time: %f milliseconds\n", milliseconds);

    // Free allocated memory
    free(visited_host);

    return 0;
}
