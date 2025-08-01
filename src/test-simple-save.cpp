#include "llama-kv-offloading.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>

int main() {
    std::cout << "=== KV Cache SSD Offloading Test ===" << std::endl;
    
    // Initialize offloader
    auto* offloader = llama_kv_offloader_init("./test_cache");
    if (!offloader) {
        std::cout << "Failed to initialize offloader" << std::endl;
        return 1;
    }
    
    std::cout << "Offloader initialized successfully" << std::endl;
    
    // Create test data
    const size_t data_size = 1024 * 1024;  // 1MB per tensor
    std::vector<float> k_data(data_size / sizeof(float));
    std::vector<float> v_data(data_size / sizeof(float));
    
    // Fill with test pattern
    for (size_t i = 0; i < k_data.size(); ++i) {
        k_data[i] = static_cast<float>(i % 256);
        v_data[i] = static_cast<float>((i * 2) % 256);
    }
    
    std::cout << "Test data created: " << data_size << " bytes per tensor" << std::endl;
    
    // Test async saving multiple layers
    const uint32_t num_layers = 5;
    auto start_time = std::chrono::steady_clock::now();
    
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        // Modify data slightly for each layer
        for (size_t i = 0; i < k_data.size(); ++i) {
            k_data[i] += layer;
            v_data[i] += layer;
        }
        
        bool success = llama_kv_offloader_save_layer(
            offloader, layer, k_data.data(), v_data.data(), data_size * 2);
        
        if (success) {
            std::cout << "Submitted save task for layer " << layer << std::endl;
        } else {
            std::cout << "Failed to submit save task for layer " << layer << std::endl;
        }
    }
    
    auto submit_time = std::chrono::steady_clock::now();
    auto submit_duration = std::chrono::duration_cast<std::chrono::milliseconds>(submit_time - start_time);
    
    std::cout << "All save tasks submitted in " << submit_duration.count() << " ms" << std::endl;
    std::cout << "Pending saves: " << llama_kv_offloader_pending_saves(offloader) << std::endl;
    
    // Wait for all saves to complete
    std::cout << "Waiting for all saves to complete..." << std::endl;
    llama_kv_offloader_wait_all(offloader);
    
    auto end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "All saves completed in " << total_duration.count() << " ms" << std::endl;
    std::cout << "Pending saves: " << llama_kv_offloader_pending_saves(offloader) << std::endl;
    
    // Calculate throughput
    size_t total_bytes = num_layers * data_size * 2;  // K + V for each layer
    double throughput_mbps = (total_bytes / 1024.0 / 1024.0) / (total_duration.count() / 1000.0);
    
    std::cout << "Total data saved: " << total_bytes / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Throughput: " << throughput_mbps << " MB/s" << std::endl;
    
    // Cleanup
    llama_kv_offloader_free(offloader);
    std::cout << "Test completed successfully!" << std::endl;
    
    return 0;
} 