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

// 테스트용
#include <unordered_map>

// =============================================================================
// Simple Task Definition
// =============================================================================

enum task_type {
    TASK_SAVE,
    TASK_LOAD,
    TASK_UPDATE_SIZE
};

struct kv_task {
    task_type type;
    uint32_t layer_id;
    
    // Save용 데이터
    void* k_data;
    void* v_data;
    size_t k_size;
    size_t v_size;

    // Load용 데이터
    ggml_tensor* k_tensor;
    ggml_tensor* v_tensor;
    
    kv_task() : type(TASK_SAVE), layer_id(0), k_data(nullptr), v_data(nullptr), k_size(0), v_size(0), 
               k_tensor(nullptr), v_tensor(nullptr) {}
};

// =============================================================================
// Main Implementation
// =============================================================================

struct llama_kv_offloader {
    std::string cache_dir;

    int is_first;
    
    // 현재 load에 사용할 KV size (worker thread에서 관리)
    size_t current_load_k_size;
    size_t current_load_v_size;
    
    // Threading
    std::thread worker_thread;
    std::atomic<bool> shutdown_flag;
    
    // Task queue
    std::queue<kv_task> task_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    
    // Counters
    std::atomic<uint32_t> pending_saves;
    std::atomic<uint32_t> pending_loads;
    std::condition_variable completion_cv;
    std::mutex completion_mutex;
    
    // Prefetch 완료 추적
    std::unordered_set<uint32_t> completed_prefetches;  // 완료된 layer들
    std::mutex prefetch_mutex;

    // === Save 데이터 보관 (검증용) ===
    struct saved_data {
        void* k_data;
        void* v_data;
        size_t k_size;
        size_t v_size;
    };
    std::unordered_map<uint32_t, saved_data> saved_layers;  // layer_id -> saved_data
    std::mutex saved_data_mutex;

    // test 용
    int iteration;
    double load_time;
    std::vector<double> load_times;

    
    llama_kv_offloader(const std::string& dir) 
        : cache_dir(dir), is_first(1), current_load_k_size(0), current_load_v_size(0),
        shutdown_flag(false), pending_saves(0), pending_loads(0), 
        iteration(0), load_time(0) {
        
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
    
    bool save_layer(uint32_t layer_id, ggml_tensor* k_tensor, ggml_tensor* v_tensor) {
        if (shutdown_flag.load()) {
            return false;
        }
        
        // Allocate and copy data
        void* k_copy = malloc(ggml_nbytes(k_tensor));
        void* v_copy = malloc(ggml_nbytes(v_tensor));
        
        if (!k_copy || !v_copy || ggml_nbytes(k_tensor) == 0 || ggml_nbytes(v_tensor) == 0) {
            free(k_copy);
            free(v_copy);
            printf("[KV-SSD] ✗ Failed to allocate memory for layer %d\n", layer_id);
            return false;
        }
        
        ggml_backend_tensor_get(k_tensor, k_copy, 0, ggml_nbytes(k_tensor));
        ggml_backend_tensor_get(v_tensor, v_copy, 0, ggml_nbytes(v_tensor));

        // Create save task
        kv_task task;
        task.type = TASK_SAVE;
        task.layer_id = layer_id;
        task.k_data = k_copy;
        task.v_data = v_copy;
        task.k_size = ggml_nbytes(k_tensor);
        task.v_size = ggml_nbytes(v_tensor);
        
        // Submit to queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.push(task);
            pending_saves.fetch_add(1);
        }
        
        queue_cv.notify_one();
        return true;
    }
    
    bool load_layer(uint32_t layer_id, ggml_tensor* k_tensor, ggml_tensor* v_tensor) {
        if (shutdown_flag.load()) {
            return false;
        }
        
        // Create load task (크기는 worker thread에서 current_load_size 사용)
        kv_task task;
        task.type = TASK_LOAD;
        task.layer_id = layer_id;
        task.k_tensor = k_tensor;
        task.v_tensor = v_tensor;
        
        // Submit to queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.push(task);
            pending_loads.fetch_add(1);
        }
        
        queue_cv.notify_one();
        return true;
    }
    
    bool update_load_size(size_t new_k_size, size_t new_v_size) {
        if (shutdown_flag.load()) {
            return false;
        }
        
        // Create size update task
        kv_task task;
        task.type = TASK_UPDATE_SIZE;
        task.k_size = new_k_size;
        task.v_size = new_v_size;
        
        // Submit to queue (최우선으로 처리되도록)
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.push(task);
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

    void wait_saves() {
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_cv.wait(lock, [this] {
            return pending_saves.load() == 0;
        });
    }
    
    bool is_prefetch_complete(uint32_t layer_id) {
        std::lock_guard<std::mutex> lock(prefetch_mutex);
        return completed_prefetches.find(layer_id) != completed_prefetches.end();
    }
    
    void reset_prefetch_status(uint32_t layer_id) {
        std::lock_guard<std::mutex> lock(prefetch_mutex);
        completed_prefetches.erase(layer_id);
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
                // 시간 측정
                const auto t_load_start = ggml_time_us();
                
                execute_load(task);
                pending_loads.fetch_sub(1);
                
                // Mark prefetch as completed
                {
                    std::lock_guard<std::mutex> lock(prefetch_mutex);
                    completed_prefetches.insert(task.layer_id);
                }

                const auto t_load_end = ggml_time_us();
                load_time += (t_load_end - t_load_start) / 1000.0;
                iteration++;
                
                if (iteration == 32) {
                    // printf("task.layer_id: %d, load_time: %f\n", task.layer_id, load_time / 32);
                    load_times.push_back(load_time);
                    iteration = 0;
                    load_time = 0;
                }
            } else if (task.type == TASK_UPDATE_SIZE) {
                execute_update_size(task);
            }
            
            completion_cv.notify_all();
        }
    }
    
    // === DATA load 시 데이터 무결성 검증 (이후 제거 예정) ===
    bool verify_data_integrity(const kv_task& task, void* k_loaded, void* v_loaded, size_t k_size_loaded, size_t v_size_loaded) {

        size_t k_size = ggml_nbytes(task.k_tensor);
        size_t v_size = ggml_nbytes(task.v_tensor);

        void * k_copy = malloc(k_size);
        void * v_copy = malloc(v_size);

        ggml_backend_tensor_get(task.k_tensor, k_copy, 0, k_size);
        ggml_backend_tensor_get(task.v_tensor, v_copy, 0, v_size);

        const char* original_k = static_cast<const char*>(k_copy);
        const char* loaded_k = static_cast<const char*>(k_loaded);

        const char* original_v = static_cast<const char*>(v_copy);
        const char* loaded_v = static_cast<const char*>(v_loaded);

        if (!k_copy || !v_copy || !k_loaded || !v_loaded) {
            printf("[KV-SSD] ⚠ Cannot verify data integrity: null pointers\n");
            return false;
        }
        
        size_t k_mismatches = 0;
        size_t v_mismatches = 0;
        
        // Verify K tensor data integrity
        if (k_size > 0) {
            
            
            for (size_t i = 0; i < k_size_loaded; ++i) {
                if (original_k[i] != loaded_k[i]) {
                    k_mismatches++;
                    if (k_mismatches <= 5) {  // Log first 5 mismatches only
                        printf("[KV-SSD] ⚠ K data mismatch at byte %zu: GPU_current=0x%02X, file_loaded=0x%02X\n", 
                               i, (unsigned char)original_k[i], (unsigned char)loaded_k[i]);
                    }
                }
            }
        }
        
        // Verify V tensor data integrity
        if (v_size > 0) {
            for (size_t i = 0; i < v_size_loaded; ++i) {
                if (original_v[i] != loaded_v[i]) {
                    v_mismatches++;
                    if (v_mismatches <= 5) {  // Log first 5 mismatches only
                        printf("[KV-SSD] ⚠ V data mismatch at byte %zu: GPU_current=0x%02X, file_loaded=0x%02X\n", 
                               i, (unsigned char)original_v[i], (unsigned char)loaded_v[i]);
                    }
                }
            }
        }
        
        // Summary report
        bool data_matches = (k_mismatches == 0 && v_mismatches == 0);
        
        if (data_matches) {
            // printf("[KV-SSD] ✓ Data integrity verified for layer %u: K=%zu bytes, V=%zu bytes (PERFECT MATCH)\n", 
            //        task.layer_id, k_size, v_size);
        } else {
            printf("[KV-SSD] ✗ Data integrity FAILED for layer %u: K_mismatches=%zu/%zu, V_mismatches=%zu/%zu\n", 
                   task.layer_id, k_mismatches, k_size_loaded, v_mismatches, v_size_loaded);
                   
            // === DEBUG: GPU 현재 데이터 샘플 출력 ===
            const unsigned char* gpu_k_bytes = static_cast<const unsigned char*>(k_copy);
            const unsigned char* gpu_v_bytes = static_cast<const unsigned char*>(v_copy);
            
            printf("[KV-SSD] 🖥️  GPU  Layer %d K[0-15]: %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X\n", 
                   task.layer_id,
                   gpu_k_bytes[0], gpu_k_bytes[1], gpu_k_bytes[2], gpu_k_bytes[3],
                   gpu_k_bytes[4], gpu_k_bytes[5], gpu_k_bytes[6], gpu_k_bytes[7],
                   gpu_k_bytes[8], gpu_k_bytes[9], gpu_k_bytes[10], gpu_k_bytes[11],
                   gpu_k_bytes[12], gpu_k_bytes[13], gpu_k_bytes[14], gpu_k_bytes[15]);
                   
            printf("[KV-SSD] 🖥️  GPU  Layer %d V[0-15]: %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X\n", 
                   task.layer_id,
                   gpu_v_bytes[0], gpu_v_bytes[1], gpu_v_bytes[2], gpu_v_bytes[3],
                   gpu_v_bytes[4], gpu_v_bytes[5], gpu_v_bytes[6], gpu_v_bytes[7],
                   gpu_v_bytes[8], gpu_v_bytes[9], gpu_v_bytes[10], gpu_v_bytes[11],
                   gpu_v_bytes[12], gpu_v_bytes[13], gpu_v_bytes[14], gpu_v_bytes[15]);
        }


        free(k_copy);
        free(v_copy);
        
        return data_matches;
        }
    
    // === SAVE-LOAD 무결성 검증 함수 === // 이 코드를 사용하기 위해서는 save 동작에서 copy 부분을 free 하지 않도록 해야함!!!! & 저장
    void verify_save_load_integrity(uint32_t layer_id, void* k_loaded, void* v_loaded, size_t k_size_loaded, size_t v_size_loaded) {
        std::lock_guard<std::mutex> lock(saved_data_mutex);
        
        auto it = saved_layers.find(layer_id);
        if (it == saved_layers.end()) {
            printf("[KV-SSD] ⚠ Layer %d: No saved data found for verification\n", layer_id);
            return;
        }
        
        const saved_data& saved = it->second;
        
        // 크기 검증
        if (saved.k_size != k_size_loaded || saved.v_size != v_size_loaded) {
            printf("[KV-SSD] ✗ Layer %d SIZE MISMATCH: saved K=%zu V=%zu, loaded K=%zu V=%zu\n", 
                   layer_id, saved.k_size, saved.v_size, k_size_loaded, v_size_loaded);
            return;
        }
        
        // K data 전체 비교
        size_t k_mismatches = 0;
        const char* saved_k = static_cast<const char*>(saved.k_data);
        const char* loaded_k = static_cast<const char*>(k_loaded);
        
        for (size_t i = 0; i < saved.k_size; ++i) {
            if (saved_k[i] != loaded_k[i]) {
                k_mismatches++;
                if (k_mismatches <= 5) {  // 처음 5개만 로깅
                    printf("[KV-SSD] ⚠ Layer %d K mismatch at byte %zu: saved=0x%02X, loaded=0x%02X\n", 
                           layer_id, i, (unsigned char)saved_k[i], (unsigned char)loaded_k[i]);
                }
            }
        }
        
        // V data 전체 비교  
        size_t v_mismatches = 0;
        const char* saved_v = static_cast<const char*>(saved.v_data);
        const char* loaded_v = static_cast<const char*>(v_loaded);
        
        for (size_t i = 0; i < saved.v_size; ++i) {
            if (saved_v[i] != loaded_v[i]) {
                v_mismatches++;
                if (v_mismatches <= 5) {  // 처음 5개만 로깅
                    printf("[KV-SSD] ⚠ Layer %d V mismatch at byte %zu: saved=0x%02X, loaded=0x%02X\n", 
                           layer_id, i, (unsigned char)saved_v[i], (unsigned char)loaded_v[i]);
                }
            }
        }
        
        // 결과 출력
        if (k_mismatches == 0 && v_mismatches == 0) {
            printf("[KV-SSD] ✓ Layer %d SAVE-LOAD integrity verified: K=%zu V=%zu bytes (PERFECT MATCH)\n", 
                   layer_id, saved.k_size, saved.v_size);
        } else {
            printf("[KV-SSD] ✗ Layer %d SAVE-LOAD integrity FAILED: K_mismatches=%zu/%zu, V_mismatches=%zu/%zu\n", 
                   layer_id, k_mismatches, saved.k_size, v_mismatches, saved.v_size);
        }
        
        // 검증 완료 후 saved data 정리
        free(saved.k_data);
        free(saved.v_data);
        saved_layers.erase(it);
    }
    
    void execute_save(const kv_task& task) {
        std::string filename = cache_dir + "/layer_" + std::to_string(task.layer_id) + ".bin";
        std::string temp_filename = filename + ".tmp";
        
        // Write to temp file
        // 시간 측정
        // const auto t_save_start = ggml_time_us();

        std::ofstream file(temp_filename, std::ios::binary);
        if (file.is_open()) {
            // === DEBUG: Save 시점 데이터 샘플 로깅 ===
            // const unsigned char* k_data_bytes = static_cast<const unsigned char*>(task.k_data);
            // const unsigned char* v_data_bytes = static_cast<const unsigned char*>(task.v_data);
            
            // printf("[KV-SSD] 💾 SAVE Layer %d K[0-15]: %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X\n", 
            //        task.layer_id, 
            //        k_data_bytes[0], k_data_bytes[1], k_data_bytes[2], k_data_bytes[3],
            //        k_data_bytes[4], k_data_bytes[5], k_data_bytes[6], k_data_bytes[7],
            //        k_data_bytes[8], k_data_bytes[9], k_data_bytes[10], k_data_bytes[11],
            //        k_data_bytes[12], k_data_bytes[13], k_data_bytes[14], k_data_bytes[15]);
                   
            // printf("[KV-SSD] 💾 SAVE Layer %d V[0-15]: %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X\n", 
            //        task.layer_id,
            //        v_data_bytes[0], v_data_bytes[1], v_data_bytes[2], v_data_bytes[3],
            //        v_data_bytes[4], v_data_bytes[5], v_data_bytes[6], v_data_bytes[7],
            //        v_data_bytes[8], v_data_bytes[9], v_data_bytes[10], v_data_bytes[11],
            //        v_data_bytes[12], v_data_bytes[13], v_data_bytes[14], v_data_bytes[15]);
            
            // Write K data
            file.write(static_cast<const char*>(task.k_data), task.k_size);
            // Write V data  
            file.write(static_cast<const char*>(task.v_data), task.v_size);
            file.close();

            // 시간 측정
            // const auto t_save_end = ggml_time_us();
            // iteration++;
            // save_time += (t_save_end - t_save_start) / 1000.0;

            // if (iteration == 32) {
            //     printf("%d, %f\n", task.layer_id, save_time / 32);
            //     iteration = 0;
            //     save_time = 0;
            // }

            // Atomic rename
            std::filesystem::rename(temp_filename, filename);
            
            // === Save 데이터 보관 (Load 시점에서 검증하기 위함) ===
            // {
            //     std::lock_guard<std::mutex> lock(saved_data_mutex);
            //     saved_data data;
            //     data.k_data = task.k_data;  // 소유권 이전 (free 안함)
            //     data.v_data = task.v_data;  // 소유권 이전 (free 안함)
            //     data.k_size = task.k_size;
            //     data.v_size = task.v_size;
            //     saved_layers[task.layer_id] = data;
            // }
            
            // printf("[KV-SSD] ✓ Saved layer %d to SSD (%zu + %zu bytes)\n", 
            //        task.layer_id, task.k_size, task.v_size);
        
            // Cleanup // 검증 코드 사용 시 주석 처리해야함!!
            free(task.k_data);
            free(task.v_data);

        } else {
            // 파일 쓰기 실패 시에만 cleanup
            free(task.k_data);
            free(task.v_data);
        }
    }
    
    void execute_update_size(const kv_task& task) {
        // Worker thread에서 load 크기 업데이트
        current_load_k_size = task.k_size;
        current_load_v_size = task.v_size;
        
        // printf("[KV-SSD] 🔄 Updated load size: K=%zu, V=%zu (total=%zu)\n", 
        //        current_load_k_size, current_load_v_size, current_load_k_size + current_load_v_size);
    }
    
    void execute_load(const kv_task& task) {
        reset_prefetch_status(task.layer_id);
        
        // Worker thread에서 current_load_size 사용
        size_t k_size = current_load_k_size;
        size_t v_size = current_load_v_size;
        
        void * k_copy = malloc(k_size);
        void * v_copy = malloc(v_size);

        if (!k_copy || !v_copy) {
            free(k_copy);
            free(v_copy);
            printf("[KV-SSD] ✗ Failed to allocate memory for layer %d\n", task.layer_id);
            return;
        }

        std::string filename = cache_dir + "/layer_" + std::to_string(task.layer_id) + ".bin";
        
        // Check if file exists
        if (!std::filesystem::exists(filename)) {
            printf("[KV-SSD] ⚠ Layer %d: File does not exist: %s\n", task.layer_id, filename.c_str());
            free(k_copy);
            free(v_copy);
            return;
        }

        // 🔍 DEBUG: 파일 크기 확인
        std::error_code ec;
        auto file_size = std::filesystem::file_size(filename, ec);
        if (ec) {
            printf("[KV-SSD] ✗ Cannot get file size for %s: %s\n", filename.c_str(), ec.message().c_str());
            free(k_copy);
            free(v_copy);
            return;
        }
        
        size_t expected_size = k_size + v_size;
        
        if (file_size != expected_size) {
            printf("[KV-SSD] ⚠ MISMATCH Layer %d: File=%zu, Expected=%zu (diff=%ld)\n", 
                   task.layer_id, file_size, expected_size, (long)(file_size - expected_size));
            // printf("[KV-SSD] ⚠ MISMATCH Layer %d: debug info removed\n", task.layer_id);
            printf("[KV-SSD] ⚠ MISMATCH Layer %d: current K=%zu, V=%zu\n",
                   task.layer_id, ggml_nbytes(task.k_tensor), ggml_nbytes(task.v_tensor));
        }
        
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            printf("[KV-SSD] ✗ Failed to open layer %d file\n", task.layer_id);
            free(k_copy);
            free(v_copy);
            return;
        }
        
        try {
            // Read K data
            
            file.read(static_cast<char*>(k_copy), k_size);
            if (file.fail()) {
                printf("[KV-SSD] ✗ Failed to read K data for layer %d\n", task.layer_id);
                file.close();
                free(k_copy);
                free(v_copy);
                return;
            }
            
            
            // Read V data
            file.read(static_cast<char*>(v_copy), v_size);
            if (file.fail()) {
                printf("[KV-SSD] ✗ Failed to read V data for layer %d\n", task.layer_id);
                file.close();
                free(k_copy);
                free(v_copy);
                return;
            }
            
            // === 확인 코드 (이후 제거 예정) ===
            // bool data_matches = verify_data_integrity(task, k_copy, v_copy, k_size, v_size);
            
            
            // printf("[KV-SSD] 💿 LOAD Layer %d K[0-15]: %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X\n", 
            //        task.layer_id,
            //        k_loaded_bytes[0], k_loaded_bytes[1], k_loaded_bytes[2], k_loaded_bytes[3],
            //        k_loaded_bytes[4], k_loaded_bytes[5], k_loaded_bytes[6], k_loaded_bytes[7],
            //        k_loaded_bytes[8], k_loaded_bytes[9], k_loaded_bytes[10], k_loaded_bytes[11],
            //        k_loaded_bytes[12], k_loaded_bytes[13], k_loaded_bytes[14], k_loaded_bytes[15]);
                   
            // printf("[KV-SSD] 💿 LOAD Layer %d V[0-15]: %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X\n", 
            //        task.layer_id,
            //        v_loaded_bytes[0], v_loaded_bytes[1], v_loaded_bytes[2], v_loaded_bytes[3],
            //        v_loaded_bytes[4], v_loaded_bytes[5], v_loaded_bytes[6], v_loaded_bytes[7],
            //        v_loaded_bytes[8], v_loaded_bytes[9], v_loaded_bytes[10], v_loaded_bytes[11],
            //        v_loaded_bytes[12], v_loaded_bytes[13], v_loaded_bytes[14], v_loaded_bytes[15]);

            // === SAVE-LOAD 데이터 무결성 검증 ===
            // verify_save_load_integrity(task.layer_id, k_copy, v_copy, k_size, v_size);
            
            ggml_backend_tensor_set(task.k_tensor, k_copy, 0, k_size);
            ggml_backend_tensor_set(task.v_tensor, v_copy, 0, v_size);

            // printf("[KV-SSD] ✓ Loaded layer %d to GPU (%zu + %zu bytes)\n", 
            //        task.layer_id, k_size, v_size);

            free(k_copy);
            free(v_copy);

            file.close();

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

void llama_kv_offloader_wait_all(struct llama_kv_offloader* offloader) {
    if (offloader) {
        offloader->wait_all();
    }
}

uint32_t llama_kv_offloader_pending_saves(struct llama_kv_offloader* offloader) {
    if (!offloader) return 0;
    return offloader->get_pending_saves();
}

// === 전체 동기화 -> 이후 필요하면 특정 작업 대기 추가 ===
void llama_kv_offloader_synchronize_all(struct llama_kv_offloader* offloader) {
    if (!offloader) return;
    offloader->wait_all();
}

void llama_kv_offloader_wait_saves(struct llama_kv_offloader* offloader) {
    if (!offloader) return;
    offloader->wait_saves();
}

void llama_kv_offloader_wait_loads(struct llama_kv_offloader* offloader) {
    if (!offloader) return;
    offloader->wait_loads();
}

bool llama_kv_offloader_save_layer(
    struct llama_kv_offloader* offloader,
    uint32_t layer_id,
    ggml_tensor* k_tensor,
    ggml_tensor* v_tensor) {
        if (!offloader) return false;
        return offloader->save_layer(layer_id, k_tensor, v_tensor);
    }

bool llama_kv_offloader_load_layer(
    struct llama_kv_offloader* offloader,
    uint32_t layer_id,
    ggml_tensor* k_tensor,
    ggml_tensor* v_tensor) {
        if (!offloader) return false;
        return offloader->load_layer(layer_id, k_tensor, v_tensor);
    }

bool llama_kv_offloader_update_load_size(
    struct llama_kv_offloader* offloader,
    size_t k_size,
    size_t v_size) {
        if (!offloader) return false;
        return offloader->update_load_size(k_size, v_size);
    }

// C++ 버전: 벡터 참조 직접 반환
const std::vector<double>& llama_kv_offloader_get_load_times(struct llama_kv_offloader* offloader) {
    static std::vector<double> empty_vector;
    
    if (!offloader) {
        return empty_vector;
    }
    
    return offloader->load_times;
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
    
    if (!cb_data || !cb_data->offloader || !tensor || !ask) {
        return true; // Continue processing
    }
    
    if (tensor->name && std::strstr(tensor->name, "kv_sync") != nullptr) {
        const char* dash_pos = std::strrchr(tensor->name, '-');
        if (dash_pos) {
            int sync_layer_id = std::atoi(dash_pos + 1);
            if (cb_data->offloader->is_first == 0) { 

                // 🎯 해당 layer의 load 완료 확인
                if (!cb_data->offloader->is_prefetch_complete(sync_layer_id)) {
                    llama_kv_offloader_wait_loads(cb_data->offloader);
                }
                
            } else if (sync_layer_id == 31) {
                cb_data->offloader->is_first = 0;
            }
        } 
        
        return true;
    }
    
    if (!tensor->name) {
        return true; // Continue processing
    }
    
    bool has_next = std::strstr(tensor->name, "_next") != nullptr;
    
    // Layer ID 추출 (공통)
    int layer_id = -1;
    bool is_k = false, is_v = false;
    
    if (has_next) {
        // load 텐서 처리
        const char* dash_pos = std::strrchr(tensor->name, '-');
        if (dash_pos) {
            layer_id = std::atoi(dash_pos + 1);
        }
        is_k = std::strncmp(tensor->name, "k_cache", 7) == 0;
        is_v = !is_k; 
    } else {
        // save 텐서 처리
        if (std::strncmp(tensor->name, "k_cache-", 8) == 0) {
            is_k = true;
            layer_id = std::atoi(tensor->name + 8);
        } else if (std::strncmp(tensor->name, "v_cache-", 8) == 0) {
            is_v = true;
            layer_id = std::atoi(tensor->name + 8);
        }
    }
    
    if ((is_k || is_v) && layer_id >= 0 && layer_id < 32) {
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
            
            // Layer 0에서 load size update task 전송 (매번)
            if (cb_data->layer_id == 0 && has_next) {
                size_t current_k_size = ggml_nbytes(cb_data->k_tensor);
                size_t current_v_size = ggml_nbytes(cb_data->v_tensor);
                
                // 항상 현재 크기로 update (FIFO 보장으로 모든 load task보다 먼저 처리됨)
                cb_data->offloader->update_load_size(current_k_size, current_v_size);
            }

            if (has_next) {
                // 🔵 Load 실행 
                if (cb_data->offloader->get_pending_loads() > 31) {
                    llama_kv_offloader_wait_loads(cb_data->offloader);
                }

                cb_data->offloader->load_layer(
                    cb_data->layer_id,
                    cb_data->k_tensor,
                    cb_data->v_tensor
                );
            } else {
                // 🟢 Save 실행
                if (cb_data->offloader->get_pending_saves() > 31) {
                    llama_kv_offloader_wait_saves(cb_data->offloader);
                }
                cb_data->offloader->save_layer(
                    cb_data->layer_id,
                    cb_data->k_tensor,
                    cb_data->v_tensor
                );
                llama_kv_offloader_wait_saves(cb_data->offloader);
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
