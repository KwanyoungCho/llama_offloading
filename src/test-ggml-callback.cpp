#include "llama-kv-offloading.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Mock ggml_tensor for testing - no GGML dependencies
struct mock_tensor {
    char name[32];
    size_t size;
    void* data;
};

int main() {
    printf("=== GGML Backend Scheduler Callback Test ===\n");
    
    // 1. KV Offloader 초기화
    printf("\n1. Initializing KV offloader...\n");
    struct llama_kv_offloader* offloader = llama_kv_offloader_init("./test_kv_cache");
    if (!offloader) {
        printf("ERROR: Failed to initialize KV offloader\n");
        return 1;
    }
    printf("✓ KV offloader initialized\n");
    
    // 2. Callback 데이터 구조 설정
    printf("\n2. Setting up callback data structure...\n");
    struct llama_kv_callback_data cb_data = {};
    cb_data.offloader = offloader;
    cb_data.current_layer = 3; // Layer 3 테스트 (2 이상이므로 저장됨)
    cb_data.k_cache_data = nullptr;
    cb_data.v_cache_data = nullptr;
    cb_data.k_cache_size = 0;
    cb_data.v_cache_size = 0;
    cb_data.k_cache_ready = false;
    cb_data.v_cache_ready = false;
    printf("✓ Callback data structure initialized for layer %d\n", cb_data.current_layer);
    
    // 3. Mock 데이터 생성
    printf("\n3. Creating mock KV cache data...\n");
    size_t k_size = 1024; // 1KB 테스트 데이터
    void* k_data = malloc(k_size);
    memset(k_data, 0xAA, k_size); // 패턴으로 채움
    
    size_t v_size = 512; // 512B 테스트 데이터
    void* v_data = malloc(v_size);
    memset(v_data, 0xBB, v_size); // 다른 패턴으로 채움
    
    printf("✓ Mock data created (K: %zu bytes, V: %zu bytes)\n", k_size, v_size);
    
    // 4. KV cache 데이터를 callback data에 설정 (실제 callback 시뮬레이션)
    printf("\n4. Simulating callback data extraction...\n");
    
    // K cache 시뮬레이션
    cb_data.k_cache_size = k_size;
    cb_data.k_cache_data = malloc(k_size);
    memcpy(cb_data.k_cache_data, k_data, k_size);
    cb_data.k_cache_ready = true;
    printf("[MOCK] K cache extracted: layer %d, size %zu bytes\n", 
           cb_data.current_layer, cb_data.k_cache_size);
    
    // V cache 시뮬레이션
    cb_data.v_cache_size = v_size;
    cb_data.v_cache_data = malloc(v_size);
    memcpy(cb_data.v_cache_data, v_data, v_size);
    cb_data.v_cache_ready = true;
    printf("[MOCK] V cache extracted: layer %d, size %zu bytes\n", 
           cb_data.current_layer, cb_data.v_cache_size);
    
    // 5. 저장 로직 테스트
    printf("\n5. Testing save logic...\n");
    if (cb_data.k_cache_ready && cb_data.v_cache_ready) {
        if (llama_kv_offloader_pending_saves(offloader) < 5) {  // Removed layer >= 2 restriction
            printf("[KV-SSD] Saving layer %d (K: %zu + V: %zu bytes)\n", 
                   cb_data.current_layer, cb_data.k_cache_size, cb_data.v_cache_size);
            
            bool success = llama_kv_offloader_save_layer(
                cb_data.offloader,
                cb_data.current_layer,
                cb_data.k_cache_data,
                cb_data.v_cache_data,
                cb_data.k_cache_size + cb_data.v_cache_size
            );
            
            if (success) {
                printf("✓ Layer %d save task submitted (pending: %u)\n", 
                       cb_data.current_layer, llama_kv_offloader_pending_saves(offloader));
            } else {
                printf("✗ Failed to submit save task for layer %d\n", cb_data.current_layer);
            }
        }
        
        // 메모리 정리
        free(cb_data.k_cache_data);
        free(cb_data.v_cache_data);
    }
    
    // 6. 추가 레이어 테스트 (layer 0 - 이제 저장됨)
    printf("\n6. Testing save logic with layer 0 (should now be saved)...\n");
    cb_data.current_layer = 0;
    cb_data.k_cache_size = k_size;
    cb_data.v_cache_size = v_size;
    cb_data.k_cache_data = malloc(k_size);
    cb_data.v_cache_data = malloc(v_size);
    memcpy(cb_data.k_cache_data, k_data, k_size);
    memcpy(cb_data.v_cache_data, v_data, v_size);
    cb_data.k_cache_ready = true;
    cb_data.v_cache_ready = true;
    
    if (cb_data.k_cache_ready && cb_data.v_cache_ready) {
        if (llama_kv_offloader_pending_saves(offloader) < 5) {  // Removed layer >= 2 restriction
            printf("[KV-SSD] Saving layer %d (K: %zu + V: %zu bytes)\n", 
                   cb_data.current_layer, cb_data.k_cache_size, cb_data.v_cache_size);
            
            bool success = llama_kv_offloader_save_layer(
                cb_data.offloader,
                cb_data.current_layer,
                cb_data.k_cache_data,
                cb_data.v_cache_data,
                cb_data.k_cache_size + cb_data.v_cache_size
            );
            
            if (success) {
                printf("✓ Layer %d save task submitted (pending: %u)\n", 
                       cb_data.current_layer, llama_kv_offloader_pending_saves(offloader));
            } else {
                printf("✗ Failed to submit save task for layer %d\n", cb_data.current_layer);
            }
        } else {
            printf("[KV-SSD] ✓ Layer %d skipped due to too many pending saves: %u\n", 
                   cb_data.current_layer, llama_kv_offloader_pending_saves(offloader));
        }
        
        free(cb_data.k_cache_data);
        free(cb_data.v_cache_data);
    }
    
    // 7. 저장 완료 대기
    printf("\n7. Waiting for save completion...\n");
    llama_kv_offloader_wait_all(offloader);
    printf("✓ All saves completed\n");
    
    // 8. 정리
    printf("\n8. Cleanup...\n");
    free(k_data);
    free(v_data);
    llama_kv_offloader_free(offloader);
    printf("✓ Cleanup completed\n");
    
    printf("\n=== New GGML Backend Scheduler Callback Test completed successfully! ===\n");
    return 0;
} 