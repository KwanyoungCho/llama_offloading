/*
 * ============================================================================
 * KV Cache SSD Offloading - Test Program
 * ============================================================================
 *
 * Test the 2-layer memory strategy and basic saving functionality
 */

#include "llama-kv-offloading.h"

#ifdef __cplusplus
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <memory>
#include <stdexcept>
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

// Mock llama_context for testing
struct mock_llama_context {
    uint32_t seq_len;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    void* kv_data;
    size_t kv_size;
};

// Mock implementation of llama.cpp functions for testing
extern "C" {
    size_t llama_kv_get_size(struct llama_context * ctx) {
        auto* mock_ctx = reinterpret_cast<mock_llama_context*>(ctx);
        return mock_ctx->kv_size;
    }
    
    size_t llama_kv_get_data(struct llama_context * ctx, uint8_t * dst, size_t size) {
        auto* mock_ctx = reinterpret_cast<mock_llama_context*>(ctx);
        if (size < mock_ctx->kv_size) return 0;
        memcpy(dst, mock_ctx->kv_data, mock_ctx->kv_size);
        return mock_ctx->kv_size;
    }
}

void test_basic_initialization() {
    std::cout << "\n=== Testing Basic Initialization ===\n";
    
    // Create test cache directory
    std::string cache_dir = "/tmp/test_kv_cache";
    std::filesystem::create_directories(cache_dir);
    
    // Initialize offloader
    auto* offloader = llama_kv_offloader_init(cache_dir.c_str(), 2, nullptr);  // === LAYER-SPECIFIC INTEGRATION ===
    if (!offloader) {
        std::cout << "FAIL: Failed to initialize offloader\n";
        return;
    }
    
    std::cout << "PASS: Offloader initialized successfully\n";
    
    // Configure for test llama model
    bool config_success = llama_kv_offloader_configure(offloader, 32, 8, 128, 16);
    if (!config_success) {
        std::cout << "FAIL: Failed to configure offloader\n";
        llama_kv_offloader_free(offloader);
        return;
    }
    
    std::cout << "PASS: Offloader configured successfully\n";
    
    // Print debug info
    llama_kv_offloader_print_debug_info(offloader);
    
    // Cleanup
    llama_kv_offloader_free(offloader);
    std::cout << "PASS: Offloader freed successfully\n";
}

void test_layer_preparation() {
    std::cout << "\n=== Testing Layer Preparation ===\n";
    
    std::string cache_dir = "/tmp/test_kv_cache";
    auto* offloader = llama_kv_offloader_init(cache_dir.c_str(), 2, nullptr);  // === LAYER-SPECIFIC INTEGRATION ===
    
    if (!offloader) {
        std::cout << "FAIL: Failed to initialize offloader\n";
        return;
    }
    
    // Configure
    llama_kv_offloader_configure(offloader, 32, 8, 128, 16);
    
    // Create mock context
    mock_llama_context mock_ctx = {};
    mock_ctx.seq_len = 100;
    mock_ctx.n_head = 32;
    mock_ctx.n_head_kv = 8;
    mock_ctx.head_dim = 128;
    mock_ctx.kv_size = mock_ctx.seq_len * mock_ctx.n_head_kv * mock_ctx.head_dim * 2 * sizeof(float);
    mock_ctx.kv_data = malloc(mock_ctx.kv_size);
    
    // Fill with test data
    auto* data = static_cast<float*>(mock_ctx.kv_data);
    for (size_t i = 0; i < mock_ctx.kv_size / sizeof(float); ++i) {
        data[i] = static_cast<float>(i % 1000) / 1000.0f;
    }
    
    // Test layer preparation (should create new layer)
    bool prep_success = llama_kv_offloader_prepare_layer(offloader, 
                                                        reinterpret_cast<llama_context*>(&mock_ctx), 
                                                        0);
    if (!prep_success) {
        std::cout << "FAIL: Failed to prepare layer 0\n";
    } else {
        std::cout << "PASS: Layer 0 prepared successfully\n";
    }
    
    // Test another layer
    prep_success = llama_kv_offloader_prepare_layer(offloader, 
                                                   reinterpret_cast<llama_context*>(&mock_ctx), 
                                                   1);
    if (!prep_success) {
        std::cout << "FAIL: Failed to prepare layer 1\n";
    } else {
        std::cout << "PASS: Layer 1 prepared successfully\n";
    }
    
    // Print debug info to see memory slots
    llama_kv_offloader_print_debug_info(offloader);
    
    // Test third layer (should trigger eviction)
    prep_success = llama_kv_offloader_prepare_layer(offloader, 
                                                   reinterpret_cast<llama_context*>(&mock_ctx), 
                                                   2);
    if (!prep_success) {
        std::cout << "FAIL: Failed to prepare layer 2\n";
    } else {
        std::cout << "PASS: Layer 2 prepared successfully (with eviction)\n";
    }
    
    // Print final state
    std::cout << "\nFinal memory state:\n";
    llama_kv_offloader_print_debug_info(offloader);
    
    // Cleanup
    free(mock_ctx.kv_data);
    llama_kv_offloader_free(offloader);
    std::cout << "PASS: Test completed\n";
}

void test_synchronization() {
    std::cout << "\n=== Testing Synchronization ===\n";
    
    std::string cache_dir = "/tmp/test_kv_cache";
    auto* offloader = llama_kv_offloader_init(cache_dir.c_str(), 2, nullptr);  // === LAYER-SPECIFIC INTEGRATION ===
    
    if (!offloader) {
        std::cout << "FAIL: Failed to initialize offloader\n";
        return;
    }
    
    // Configure
    llama_kv_offloader_configure(offloader, 32, 8, 128, 16);
    
    // Test synchronization (should complete immediately with no pending ops)
    llama_kv_offloader_synchronize(offloader);
    std::cout << "PASS: Synchronization completed\n";
    
    // Check pending operations
    uint32_t pending = llama_kv_offloader_pending_operations(offloader);
    std::cout << "Pending operations: " << pending << "\n";
    
    // Cleanup
    llama_kv_offloader_free(offloader);
    std::cout << "PASS: Synchronization test completed\n";
}

void test_debug_levels() {
    std::cout << "\n=== Testing Debug Levels ===\n";
    
    // Test different debug levels
    llama_kv_offloader_set_debug_level(4);  // Debug level
    std::cout << "Set debug level to 4 (debug)\n";
    
    std::string cache_dir = "/tmp/test_kv_cache";
    auto* offloader = llama_kv_offloader_init(cache_dir.c_str(), 2, nullptr);  // === LAYER-SPECIFIC INTEGRATION ===
    
    if (offloader) {
        llama_kv_offloader_configure(offloader, 32, 8, 128, 16);
        
        // This should produce debug output
        mock_llama_context mock_ctx = {};
        mock_ctx.seq_len = 50;
        mock_ctx.n_head = 32;
        mock_ctx.n_head_kv = 8;
        mock_ctx.head_dim = 128;
        mock_ctx.kv_size = 1024;  // Small test size
        mock_ctx.kv_data = malloc(mock_ctx.kv_size);
        
        llama_kv_offloader_prepare_layer(offloader, 
                                        reinterpret_cast<llama_context*>(&mock_ctx), 
                                        0);
        
        free(mock_ctx.kv_data);
        llama_kv_offloader_free(offloader);
    }
    
    // Reset to normal level
    llama_kv_offloader_set_debug_level(1);
    std::cout << "PASS: Debug level test completed\n";
}

int main() {
    std::cout << "KV Cache SSD Offloading - Test Suite\n";
    std::cout << "====================================\n";
    
    try {
        test_basic_initialization();
        test_layer_preparation();
        test_synchronization();
        test_debug_levels();
        
        std::cout << "\n=== All Tests Completed ===\n";
        std::cout << "Check /tmp/test_kv_cache for generated files\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
} 