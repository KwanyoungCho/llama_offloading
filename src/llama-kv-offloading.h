#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <vector>
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// KV Cache SSD Offloading - C API (Minimal Implementation)
// =============================================================================

/**
 * Opaque handle for KV cache offloader
 */
 struct llama_kv_offloader;

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
  * Wait for all pending save operations to complete
  */
 void llama_kv_offloader_wait_all(struct llama_kv_offloader* offloader);
 
 /**
  * Wait for all pending load operations to complete
  */
 void llama_kv_offloader_wait_loads(struct llama_kv_offloader* offloader);
 
 /**
  * Get number of pending save operations
  */
 uint32_t llama_kv_offloader_pending_saves(struct llama_kv_offloader* offloader);
 
 /**
  * Get number of pending load operations
  */
 uint32_t llama_kv_offloader_pending_loads(struct llama_kv_offloader* offloader);
 
 /**
  * Synchronize all pending operations
  */
 void llama_kv_offloader_synchronize_all(struct llama_kv_offloader* offloader);

 /**
  * Submit an asynchronous save for layer_id with K and V data
  *
  * @return true if enqueued successfully, false on error
  */
 bool llama_kv_offloader_save_layer(
     struct llama_kv_offloader* offloader,
     uint32_t layer_id,
     ggml_tensor* k_tensor,
     ggml_tensor* v_tensor
 );
 
 /**
  * Submit an asynchronous load for layer_id into provided K and V buffers
  *
  * @return true if enqueued successfully, false on error
  */
 bool llama_kv_offloader_load_layer(
     struct llama_kv_offloader* offloader,
     uint32_t layer_id,
     ggml_tensor* k_tensor,
     ggml_tensor* v_tensor
 );

#ifdef __cplusplus
/**
 * 
 * @param offloader Offloader handle
 * @return Reference to load times vector
 */
const std::vector<double>& llama_kv_offloader_get_load_times(struct llama_kv_offloader* offloader);
#endif

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

