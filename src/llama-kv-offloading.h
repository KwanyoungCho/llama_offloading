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
 * @param data_size Total size of K + V data
 * @return true if save task submitted successfully
 */
bool llama_kv_offloader_save_layer(
    struct llama_kv_offloader* offloader,
    uint32_t layer_id,
    const void* k_data,
    const void* v_data,
    size_t data_size
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
// GGML Backend Scheduler Callback Integration (ADDED)
// =============================================================================

/**
 * Callback data structure for GGML backend scheduler
 */
struct llama_kv_callback_data {
    struct llama_kv_offloader* offloader;
    int current_layer;
    void* k_cache_data;
    void* v_cache_data;
    size_t k_cache_size;
    size_t v_cache_size;
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

