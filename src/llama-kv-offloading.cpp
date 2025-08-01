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

// =============================================================================
// Simple Task Definition
// =============================================================================

struct save_task {
    uint32_t layer_id;
    void* k_data;
    void* v_data;
    size_t data_size;
    
    save_task() : layer_id(0), k_data(nullptr), v_data(nullptr), data_size(0) {}
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
    std::queue<save_task> task_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    
    // Counters
    std::atomic<uint32_t> pending_saves;
    std::condition_variable completion_cv;
    std::mutex completion_mutex;
    
    llama_kv_offloader(const std::string& dir) 
        : cache_dir(dir), shutdown_flag(false), pending_saves(0) {
        
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
    
    bool save_layer(uint32_t layer_id, const void* k_data, const void* v_data, size_t data_size) {
        if (shutdown_flag.load()) {
            return false;
        }
        
        // Allocate and copy data
        void* k_copy = malloc(data_size / 2);
        void* v_copy = malloc(data_size / 2);
        
        if (!k_copy || !v_copy) {
            free(k_copy);
            free(v_copy);
            return false;
        }
        
        memcpy(k_copy, k_data, data_size / 2);
        memcpy(v_copy, v_data, data_size / 2);
        
        // Create task
        save_task task;
        task.layer_id = layer_id;
        task.k_data = k_copy;
        task.v_data = v_copy;
        task.data_size = data_size;
        
        // Submit to queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.push(task);
            pending_saves.fetch_add(1);
        }
        
        queue_cv.notify_one();
        return true;
    }
    
    void wait_all() {
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_cv.wait(lock, [this] {
            return pending_saves.load() == 0;
        });
    }
    
    uint32_t get_pending_saves() const {
        return pending_saves.load();
    }
    
private:
    void worker_main() {
        while (!shutdown_flag.load()) {
            save_task task;
            
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
            
            // Execute save
            execute_save(task);
            
            // Update counter and notify
            pending_saves.fetch_sub(1);
            completion_cv.notify_all();
        }
    }
    
    void execute_save(const save_task& task) {
        std::string filename = cache_dir + "/layer_" + std::to_string(task.layer_id) + ".bin";
        std::string temp_filename = filename + ".tmp";
        
        // Write to temp file
        std::ofstream file(temp_filename, std::ios::binary);
        if (file.is_open()) {
            // Write K data
            file.write(static_cast<const char*>(task.k_data), task.data_size / 2);
            // Write V data  
            file.write(static_cast<const char*>(task.v_data), task.data_size / 2);
            file.close();
            
            // Atomic rename
            std::filesystem::rename(temp_filename, filename);
        }
        
        // Cleanup
        free(task.k_data);
        free(task.v_data);
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
    size_t data_size) {
    
    if (!offloader) return false;
    return offloader->save_layer(layer_id, k_data, v_data, data_size);
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
    
    if (!cb_data || !cb_data->offloader || !tensor) {
        return true; // Continue processing
    }
    
    if (ask) {
        // 실행 전: KV cache 텐서 식별
        if (!tensor->name) {
            return false;
        }
        
        // K cache 또는 V cache 텐서인지 확인 (layer 번호 포함된 패턴)
        // 실제 name: "k_cache-0", "k_cache-1", ..., "v_cache-0", "v_cache-1", ...
        bool is_k_cache = (strncmp(tensor->name, "k_cache-", 8) == 0);
        bool is_v_cache = (strncmp(tensor->name, "v_cache-", 8) == 0);
        return is_k_cache || is_v_cache;
    }
    
    // 실행 후: 실제 데이터 추출 및 저장
    if (!tensor->name) return true;
    
    // Layer ID 추출 - tensor name에서 layer 번호 추출
    int layer_id = -1;
    if (strncmp(tensor->name, "k_cache-", 8) == 0) {
        layer_id = atoi(tensor->name + 8);  // "k_cache-" 이후 숫자 파싱
    } else if (strncmp(tensor->name, "v_cache-", 8) == 0) {
        layer_id = atoi(tensor->name + 8);  // "v_cache-" 이후 숫자 파싱
    }
    
    if (layer_id < 0) return true;  // KV cache tensor가 아니면 continue
    
    if (strncmp(tensor->name, "k_cache-", 8) == 0) {
        // K cache 데이터 추출
        size_t tensor_size = ggml_nbytes(tensor);
        void* tensor_data = ggml_get_data(tensor);
        
        if (tensor_data && tensor_size > 0) {
            cb_data->k_cache_size = tensor_size;
            cb_data->k_cache_data = malloc(tensor_size);
            if (cb_data->k_cache_data) {
                memcpy(cb_data->k_cache_data, tensor_data, tensor_size);
                cb_data->k_cache_ready = true;
                cb_data->current_layer = layer_id;
            }
        }
    }
    
    if (strncmp(tensor->name, "v_cache-", 8) == 0) {
        // V cache 데이터 추출
        size_t tensor_size = ggml_nbytes(tensor);
        void* tensor_data = ggml_get_data(tensor);
        
        if (tensor_data && tensor_size > 0) {
            cb_data->v_cache_size = tensor_size;
            cb_data->v_cache_data = malloc(tensor_size);
            if (cb_data->v_cache_data) {
                memcpy(cb_data->v_cache_data, tensor_data, tensor_size);
                cb_data->v_cache_ready = true;
                cb_data->current_layer = layer_id;
            }
        }
    }
    
    // K, V 모두 준비되면 SSD 저장
    if (cb_data->k_cache_ready && cb_data->v_cache_ready) {
        // 저장 조건 확인 (pending saves 제한만 적용)
        if (cb_data->offloader->get_pending_saves() < 5) {
            llama_kv_offloader_save_layer(
                cb_data->offloader,
                cb_data->current_layer,
                cb_data->k_cache_data,
                cb_data->v_cache_data,
                cb_data->k_cache_size + cb_data->v_cache_size
            );
        }
        
        // 메모리 정리
        free(cb_data->k_cache_data);
        free(cb_data->v_cache_data);
        cb_data->k_cache_data = nullptr;
        cb_data->v_cache_data = nullptr;
        cb_data->k_cache_ready = cb_data->v_cache_ready = false;
    }
    
    return true; // Continue processing
}

} // extern "C"
