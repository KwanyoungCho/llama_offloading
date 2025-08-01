/*
 * ============================================================================
 * KV Cache SSD Offloading - Minimal Async Save Implementation
 * ============================================================================
 */

#include "llama-kv-offloading.h"
#include "ggml.h"          // ADDED: For ggml_tensor, ggml_nbytes, ggml_get_data
#include "ggml-backend.h"  // ADDED: For ggml_backend_sched_eval_callback

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

// =============================================================================
// Simple Task Definition
// =============================================================================

enum task_type {
    TASK_SAVE,
    TASK_LOAD
};

struct kv_task {
    task_type type;
    uint32_t layer_id;
    
    // Save용 데이터
    void* k_data;
    void* v_data;
    size_t k_size;
    size_t v_size;
    
    // Load용 목적지
    void* k_dest;
    void* v_dest;
    
    kv_task() : type(TASK_SAVE), layer_id(0), k_data(nullptr), v_data(nullptr), 
                k_size(0), v_size(0), k_dest(nullptr), v_dest(nullptr) {}
};

// =============================================================================
// Main Implementation
// =============================================================================

struct llama_kv_offloader {
    std::string cache_dir;
    
    // Threading
    std::thread worker_thread;
    std::atomic<bool> shutdown_flag;
    
    // Task queue
    std::queue<kv_task> task_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    
    // Counters
    std::atomic<uint32_t> pending_saves;
    std::atomic<uint32_t> pending_loads;  // Load 작업 추적
    std::condition_variable completion_cv;
    std::mutex completion_mutex;
    
    // Prefetch 완료 추적
    std::unordered_set<uint32_t> completed_prefetches;  // 완료된 layer들
    std::mutex prefetch_mutex;
    
    llama_kv_offloader(const std::string& dir) 
        : cache_dir(dir), shutdown_flag(false), pending_saves(0), pending_loads(0) {
        
        // Create cache directory
        std::filesystem::create_directories(cache_dir);
        
        // Start worker thread
        worker_thread = std::thread(&llama_kv_offloader::worker_main, this);
    }
    
    ~llama_kv_offloader() {
        shutdown();
    }
    
    void shutdown() {
        if (!shutdown_flag.load()) {
            shutdown_flag.store(true);
            queue_cv.notify_all();
            completion_cv.notify_all();
            
            if (worker_thread.joinable()) {
                worker_thread.join();
            }
        }
    }
    
    bool save_layer(uint32_t layer_id, const void* k_data, const void* v_data, size_t k_data_size, size_t v_data_size) {
        if (shutdown_flag.load()) {
            return false;
        }
        
        // Allocate and copy data
        void* k_copy = malloc(k_data_size);
        void* v_copy = malloc(v_data_size);
        
        if (!k_copy || !v_copy) {
            free(k_copy);
            free(v_copy);
            return false;
        }
        
        memcpy(k_copy, k_data, k_data_size);
        memcpy(v_copy, v_data, v_data_size);
        
        // Create save task
        kv_task task;
        task.type = TASK_SAVE;
        task.layer_id = layer_id;
        task.k_data = k_copy;
        task.v_data = v_copy;
        task.k_size = k_data_size;
        task.v_size = v_data_size;
        
        // Submit to queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.push(task);
            pending_saves.fetch_add(1);
        }
        
        queue_cv.notify_one();
        return true;
    }
    
    bool load_layer(uint32_t layer_id, void* k_dest, void* v_dest, 
                    size_t k_size, size_t v_size) {
        if (shutdown_flag.load()) {
            return false;
        }
        
        // Create load task
        kv_task task;
        task.type = TASK_LOAD;
        task.layer_id = layer_id;
        task.k_dest = k_dest;
        task.v_dest = v_dest;
        task.k_size = k_size;
        task.v_size = v_size;
        
        // Submit to queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.push(task);
            pending_loads.fetch_add(1);
        }
        
        queue_cv.notify_one();
        return true;
    }
    
    void wait_all() {
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_cv.wait(lock, [this] {
            return pending_saves.load() == 0 && pending_loads.load() == 0;
        });
    }
    
    void wait_loads() {
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_cv.wait(lock, [this] {
            return pending_loads.load() == 0;
        });
    }
    
    bool is_prefetch_complete(uint32_t layer_id) {
        std::lock_guard<std::mutex> lock(prefetch_mutex);
        return completed_prefetches.find(layer_id) != completed_prefetches.end();
    }
    
    uint32_t get_pending_saves() const {
        return pending_saves.load();
    }
    
    uint32_t get_pending_loads() const {
        return pending_loads.load();
    }
    
private:
    void worker_main() {
        while (!shutdown_flag.load()) {
            kv_task task;
            
            // Get task from queue
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this] {
                    return !task_queue.empty() || shutdown_flag.load();
                });
                
                if (shutdown_flag.load() && task_queue.empty()) {
                    break;
                }
                
                if (!task_queue.empty()) {
                    task = task_queue.front();
                    task_queue.pop();
                } else {
                    continue;
                }
            }
            
            // Execute task based on type
            if (task.type == TASK_SAVE) {
                execute_save(task);
                pending_saves.fetch_sub(1);
            } else if (task.type == TASK_LOAD) {
                execute_load(task);
                pending_loads.fetch_sub(1);
                
                // Mark prefetch as completed
                {
                    std::lock_guard<std::mutex> lock(prefetch_mutex);
                    completed_prefetches.insert(task.layer_id);
                }
            }
            
            completion_cv.notify_all();
        }
    }
    
    void execute_save(const kv_task& task) {
        std::string filename = cache_dir + "/layer_" + std::to_string(task.layer_id) + ".bin";
        std::string temp_filename = filename + ".tmp";
        
        // Write to temp file
        std::ofstream file(temp_filename, std::ios::binary);
        if (file.is_open()) {
            // Write K data
            file.write(static_cast<const char*>(task.k_data), task.k_size);
            // Write V data  
            file.write(static_cast<const char*>(task.v_data), task.v_size);
            file.close();
            
            // Atomic rename
            std::filesystem::rename(temp_filename, filename);
            
            // printf("[KV-SSD] ✓ Saved layer %d to SSD (%zu + %zu bytes)\n", 
            //        task.layer_id, task.k_size, task.v_size);
        }
        
        // Cleanup
        free(task.k_data);
        free(task.v_data);
    }
    
    void execute_load(const kv_task& task) {
        std::string filename = cache_dir + "/layer_" + std::to_string(task.layer_id) + ".bin";
        
        // Check if file exists
        if (!std::filesystem::exists(filename)) {
            // printf("[KV-SSD] ⚠ Layer %d not found on SSD, skipping load\n", task.layer_id);
            return;
        }
        
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            printf("[KV-SSD] ✗ Failed to open layer %d file\n", task.layer_id);
            return;
        }
        
        try {
            // Read K data
            if (task.k_dest && task.k_size > 0) {
                file.read(static_cast<char*>(task.k_dest), task.k_size);
                if (file.fail()) {
                    printf("[KV-SSD] ✗ Failed to read K data for layer %d\n", task.layer_id);
                    file.close();
                    return;
                }
            }
            
            // Read V data
            if (task.v_dest && task.v_size > 0) {
                file.read(static_cast<char*>(task.v_dest), task.v_size);
                if (file.fail()) {
                    printf("[KV-SSD] ✗ Failed to read V data for layer %d\n", task.layer_id);
                    file.close();
                    return;
                }
            }
            
            file.close();
            // printf("[KV-SSD] ✓ Loaded layer %d from SSD (%zu + %zu bytes)\n", 
            //        task.layer_id, task.k_size, task.v_size);
            
        } catch (...) {
            printf("[KV-SSD] ✗ Exception during load for layer %d\n", task.layer_id);
            file.close();
        }
    }
};

// =============================================================================
// C Interface
// =============================================================================

extern "C" {

struct llama_kv_offloader* llama_kv_offloader_init(const char* cache_dir) {
    try {
        return new llama_kv_offloader(cache_dir);
    } catch (...) {
        return nullptr;
    }
}

void llama_kv_offloader_free(struct llama_kv_offloader* offloader) {
    delete offloader;
}

bool llama_kv_offloader_save_layer(
    struct llama_kv_offloader* offloader,
    uint32_t layer_id,
    const void* k_data,
    const void* v_data,
    size_t k_data_size,
    size_t v_data_size) {
    
    if (!offloader) return false;
    return offloader->save_layer(layer_id, k_data, v_data, k_data_size, v_data_size);
}

void llama_kv_offloader_wait_all(struct llama_kv_offloader* offloader) {
    if (offloader) {
        offloader->wait_all();
    }
}

uint32_t llama_kv_offloader_pending_saves(struct llama_kv_offloader* offloader) {
    if (!offloader) return 0;
    return offloader->get_pending_saves();
}

} // extern "C"

// =============================================================================
// GGML Backend Scheduler Callback Implementation (ADDED)
// =============================================================================

extern "C" {

bool llama_kv_ggml_eval_callback(
    struct ggml_tensor * tensor, 
    bool ask, 
    void * user_data) {
    
    auto* cb_data = static_cast<llama_kv_callback_data*>(user_data);
    
    if (!cb_data || !cb_data->offloader || !tensor || !ask || !tensor->name) {
        return true; // Continue processing
    }
    
    // 모든 처리를 ask=true에서만 (이미 메모리 위치가 있는 상태)
    bool has_next = std::strstr(tensor->name, "_next") != nullptr;
    
    // Layer ID 추출 (공통)
    int layer_id = -1;
    bool is_k = false, is_v = false;
    
    if (has_next) {
        // _next 텐서: dash로 구분된 layer ID (예: k_cache_next-0, v_cache_next-0)
        const char* dash_pos = std::strrchr(tensor->name, '-');
        if (dash_pos) {
            layer_id = std::atoi(dash_pos + 1);
        }
        is_k = std::strncmp(tensor->name, "k_cache", 7) == 0;
        is_v = !is_k; // k_cache가 아니면 v_cache
    } else {
        // 일반 텐서: dash로 구분된 layer ID  
        if (std::strncmp(tensor->name, "k_cache-", 8) == 0) {
            is_k = true;
            layer_id = std::atoi(tensor->name + 8);
        } else if (std::strncmp(tensor->name, "v_cache-", 8) == 0) {
            is_v = true;
            layer_id = std::atoi(tensor->name + 8);
        }
    }
    
    if ((is_k || is_v) && layer_id >= 0 && layer_id < 32) {
        // 동일한 패턴으로 텐서 수집
        if (is_k) {
            cb_data->k_tensor = tensor;
            cb_data->k_cache_ready = true;
            cb_data->layer_id = layer_id;
        } else {
            cb_data->v_tensor = tensor;
            cb_data->v_cache_ready = true;
            cb_data->layer_id = layer_id;
        }
        
        // K, V 모두 준비되면 실행 (Load 또는 Save)
        if (cb_data->k_cache_ready && cb_data->v_cache_ready) {
            if (has_next) {
                // 🔵 Load 실행
                if (cb_data->offloader->get_pending_loads() < 8) {
                    cb_data->offloader->load_layer(
                        cb_data->layer_id,
                        ggml_get_data(cb_data->k_tensor),
                        ggml_get_data(cb_data->v_tensor),
                        ggml_nbytes(cb_data->k_tensor),
                        ggml_nbytes(cb_data->v_tensor)
                    );
                    // printf("[KV-SSD] → Load task submitted for layer %d\n", cb_data->layer_id);
                }
            } else {
                // 🟢 Save 실행
                if (cb_data->offloader->get_pending_saves() < 8) {
                    cb_data->offloader->save_layer(
                        cb_data->layer_id,
                        ggml_get_data(cb_data->k_tensor),
                        ggml_get_data(cb_data->v_tensor),
                        ggml_nbytes(cb_data->k_tensor),
                        ggml_nbytes(cb_data->v_tensor)
                    );
                    // printf("[KV-SSD] → Save task submitted for layer %d\n", cb_data->layer_id);
                }
            }
            
            // Reset (공통)
            cb_data->k_tensor = nullptr;
            cb_data->v_tensor = nullptr;
            cb_data->k_cache_ready = cb_data->v_cache_ready = false;
        }
    }
    
    return true;
}



} // extern "C"
