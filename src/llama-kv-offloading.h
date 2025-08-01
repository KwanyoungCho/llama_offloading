#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// KV Cache SSD Offloading - Minimal Implementation
// =============================================================================

/**
 * Opaque handle for KV cache offloader
 */
struct llama_kv_offloader;

/**
 * Forward declaration of llama_context
 */
struct llama_context;

// =============================================================================
// Core Functions - Save Only
// =============================================================================

/**
 * Initialize KV cache offloader
 * 
 * @param cache_dir Directory for storing KV cache files
 * @return Offloader handle or NULL on failure
 */
struct llama_kv_offloader* llama_kv_offloader_init(const char* cache_dir);

/**
 * Free offloader and cleanup all resources
 */
void llama_kv_offloader_free(struct llama_kv_offloader* offloader);

/**
 * Save layer data to SSD asynchronously
 * 
 * @param offloader Offloader handle
 * @param layer_id Layer ID to save
 * @param k_data K tensor data
 * @param v_data V tensor data
 * @param k_data_size Size of K tensor data
 * @param v_data_size Size of V tensor data
 * @return true if save task submitted successfully
 */
bool llama_kv_offloader_save_layer(
    struct llama_kv_offloader* offloader,
    uint32_t layer_id,
    const void* k_data,
    const void* v_data,
    size_t k_data_size,
    size_t v_data_size
);

/**
 * Wait for all pending save operations to complete
 */
void llama_kv_offloader_wait_all(struct llama_kv_offloader* offloader);

/**
 * Get number of pending save operations
 */
uint32_t llama_kv_offloader_pending_saves(struct llama_kv_offloader* offloader);

// =============================================================================
// GGML Backend Scheduler Callback Integration
// =============================================================================

/**
 * Callback data structure for GGML backend scheduler (OPTIMIZED)
 */
struct llama_kv_callback_data {
    struct llama_kv_offloader* offloader;
    int layer_id;  // 현재 처리중인 layer ID (Load/Save 공통)
    
    // Load/Save 공통: 텐서 포인터들
    struct ggml_tensor* k_tensor;      // K cache tensor pointer
    struct ggml_tensor* v_tensor;      // V cache tensor pointer
    
    bool k_cache_ready;
    bool v_cache_ready;
};

/**
 * GGML Backend Scheduler callback for KV cache SSD saving
 * This is called during graph compute at the right timing
 * 
 * @param tensor Current tensor being processed
 * @param ask true=before execution, false=after execution
 * @param user_data Pointer to llama_kv_callback_data
 * @return true to continue, false to stop
 */
bool llama_kv_ggml_eval_callback(
    struct ggml_tensor * tensor, 
    bool ask, 
    void * user_data
);

#ifdef __cplusplus
}
#endif

