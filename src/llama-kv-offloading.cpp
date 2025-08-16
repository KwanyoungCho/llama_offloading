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

    // Delta metadata
    int k_head;
    int k_n_new;
    int v_head;
    int v_n_new;
    
    kv_task() : type(TASK_SAVE), layer_id(0), k_data(nullptr), v_data(nullptr), k_size(0), v_size(0), 
               k_tensor(nullptr), v_tensor(nullptr), k_head(0), k_n_new(0), v_head(0), v_n_new(0) {}
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
    
    bool save_layer(uint32_t layer_id, ggml_tensor* k_tensor, ggml_tensor* v_tensor,
                    int k_head, int k_n_new, int v_head, int v_n_new) {
        if (shutdown_flag.load()) {
            return false;
        }
        // Create delta save task with pre-copied CPU buffers (synchronous copy here)
        kv_task task;
        task.type = TASK_SAVE;
        task.layer_id = layer_id;
        task.k_tensor = k_tensor;
        task.v_tensor = v_tensor;
        task.k_head = k_head;
        task.k_n_new = k_n_new;
        task.v_head = v_head;
        task.v_n_new = v_n_new;
 
        // Gather tensor shapes/strides
        const int64_t k_ne0 = k_tensor->ne[0];
        const int64_t k_ne1 = k_tensor->ne[1];
        const size_t  k_nb0 = k_tensor->nb[0];
        const size_t  k_nb1 = k_tensor->nb[1];
        const size_t  k_nb2 = k_tensor->nb[2];
 
        const int64_t v_ne1 = v_tensor->ne[1];
        const int64_t v_ne2 = v_tensor->ne[2];
        const size_t  v_nb0 = v_tensor->nb[0];
        const size_t  v_nb1 = v_tensor->nb[1];
        const size_t  v_nb2 = v_tensor->nb[2];
 
        // K packed delta buffer: [h in 0..ne1) x [t in 0..k_n_new): row_bytes
        const size_t k_row_bytes = (size_t)k_ne0 * k_nb0;
        const size_t k_total_delta = (size_t)k_ne1 * (size_t)k_n_new * k_row_bytes;
        void* k_packed = nullptr;
        if (k_total_delta > 0) {
            k_packed = malloc(k_total_delta);
            if (!k_packed) return false;
            uint8_t* dst = static_cast<uint8_t*>(k_packed);
            for (int64_t h = 0; h < k_ne1; ++h) {
                for (int t = 0; t < k_n_new; ++t) {
                    const size_t tensor_off = (size_t)h * k_nb1 + (size_t)(k_head + t) * k_nb2;
                    ggml_backend_tensor_get(k_tensor, dst, tensor_off, k_row_bytes);
                    dst += k_row_bytes;
                }
            }
        }
        task.k_data = k_packed;
        task.k_size = k_total_delta;
 
        // V packed delta buffer (v_trans=true): [h in 0..ne1) x [e in 0..ne2): block_bytes across kv
        const size_t v_block_bytes = (size_t)v_n_new * v_nb0; // nb0=elsize
        const size_t v_total_delta = (size_t)v_ne1 * (size_t)v_ne2 * v_block_bytes;
        void* v_packed = nullptr;
        if (v_total_delta > 0) {
            v_packed = malloc(v_total_delta);
            if (!v_packed) {
                free(k_packed);
                return false;
            }
            uint8_t* dst = static_cast<uint8_t*>(v_packed);
            for (int64_t h = 0; h < v_ne1; ++h) {
                for (int64_t e = 0; e < v_ne2; ++e) {
                    const size_t tensor_off = (size_t)h * v_nb1 + (size_t)e * v_nb2 + (size_t)v_head * v_nb0;
                    ggml_backend_tensor_get(v_tensor, dst, tensor_off, v_block_bytes);
                    dst += v_block_bytes;
                }
            }
        }
        task.v_data = v_packed;
        task.v_size = v_total_delta;
 
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
                execute_save_delta(task);
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
                    load_times.push_back(load_time / 32.0);
                    iteration = 0;
                    load_time = 0;
                }
            }
            
            completion_cv.notify_all();
        }
    }
    
    // removed verification helpers and saved_layers usage
 
    // Lightweight verification: compare SSD-read buffers with current GPU tensor contents (first N bytes)
    bool verify_data_integrity(const kv_task& task, void* k_loaded, void* v_loaded,
                               size_t k_size_loaded, size_t v_size_loaded) {
        bool ok = true;
        if (k_loaded && k_size_loaded > 0) {
            void* k_ref = malloc(k_size_loaded);
            if (k_ref) {
                ggml_backend_tensor_get(task.k_tensor, k_ref, 0, k_size_loaded);
                if (std::memcmp(k_ref, k_loaded, k_size_loaded) != 0) {
                    printf("[KV-SSD] ✗ K data mismatch for layer %u (bytes=%zu)\n", task.layer_id, k_size_loaded);
                    ok = false;
                }
                free(k_ref);
            }
        }
        if (v_loaded && v_size_loaded > 0) {
            void* v_ref = malloc(v_size_loaded);
            if (v_ref) {
                ggml_backend_tensor_get(task.v_tensor, v_ref, 0, v_size_loaded);
                if (std::memcmp(v_ref, v_loaded, v_size_loaded) != 0) {
                    printf("[KV-SSD] ✗ V data mismatch for layer %u (bytes=%zu)\n", task.layer_id, v_size_loaded);
                    ok = false;
                }
                free(v_ref);
            }
        }
        if (ok) {
            printf("[KV-SSD] ✓ SSD vs GPU match for layer %u (K=%zu, V=%zu)\n", task.layer_id, k_size_loaded, v_size_loaded);
        }
        return ok;
    }

    void ensure_file_size(const std::string & path, size_t min_size) {
        std::error_code ec;
        if (!std::filesystem::exists(path)) {
            // create empty file
            std::ofstream f(path, std::ios::binary);
            f.close();
        }
        auto cur = std::filesystem::exists(path) ? std::filesystem::file_size(path, ec) : 0;
        if (ec || cur < min_size) {
            std::filesystem::resize_file(path, min_size, ec);
        }
    }
 
    void execute_save_delta(const kv_task& task) {
        // K/V split files
        std::string kfile = cache_dir + "/layer_" + std::to_string(task.layer_id) + "_K.bin";
        std::string vfile = cache_dir + "/layer_" + std::to_string(task.layer_id) + "_V.bin";

        // derive tensor shapes/strides
        auto * kt = task.k_tensor;
        auto * vt = task.v_tensor;
        if (!kt || !vt) return;

        const int64_t k_ne0 = kt->ne[0];
        const int64_t k_ne1 = kt->ne[1];
        const int64_t k_ne2 = kt->ne[2];
        const size_t  k_nb0 = kt->nb[0];
        const size_t  k_nb1 = kt->nb[1];
        const size_t  k_nb2 = kt->nb[2];

        const int64_t v_ne0 = vt->ne[0];
        const int64_t v_ne1 = vt->ne[1];
        const int64_t v_ne2 = vt->ne[2];
        const size_t  v_nb0 = vt->nb[0];
        const size_t  v_nb1 = vt->nb[1];
        const size_t  v_nb2 = vt->nb[2];

        // Compute minimal file size needed for full tensors (matches ggml_nbytes)
        const size_t k_total_bytes = (k_ne0 > 0 && k_ne1 > 0 && k_ne2 > 0) ? (k_ne0 - 1) * k_nb0 + (k_ne1 - 1) * k_nb1 + (k_ne2 - 1) * k_nb2 + k_nb0 : 0;
        const size_t v_total_bytes = (v_ne0 > 0 && v_ne1 > 0 && v_ne2 > 0) ? (v_ne0 - 1) * v_nb0 + (v_ne1 - 1) * v_nb1 + (v_ne2 - 1) * v_nb2 + v_nb0 : 0;

        ensure_file_size(kfile, k_total_bytes);
        ensure_file_size(vfile, v_total_bytes);

        // Open files R/W
        std::fstream kfs(kfile, std::ios::in | std::ios::out | std::ios::binary);
        std::fstream vfs(vfile, std::ios::in | std::ios::out | std::ios::binary);
        if (!kfs.is_open() || !vfs.is_open()) {
            printf("[KV-SSD] ✗ Cannot open K/V files for layer %u\n", task.layer_id);
            return;
        }
        
        // K: for each head and each new token, write one contiguous row (embd_k_per_head) from packed buffer
        const size_t k_row_bytes = (size_t) k_ne0 * k_nb0;
        const uint8_t* ksrc = static_cast<const uint8_t*>(task.k_data);
        for (int64_t h = 0; h < k_ne1; ++h) {
            for (int t = 0; t < task.k_n_new; ++t) {
                const size_t tensor_off = (size_t)h * k_nb1 + (size_t)(task.k_head + t) * k_nb2;
                kfs.seekp((std::streamoff)tensor_off, std::ios::beg);
                kfs.write(reinterpret_cast<const char*>(ksrc), (std::streamsize)k_row_bytes);
                ksrc += k_row_bytes;
                if (kfs.fail()) {
                    printf("[KV-SSD] ✗ K delta write failed: layer %u, head %ld, t %d\n", task.layer_id, (long)h, t);
            return;
        }
            }
        }

        // V: v_trans=true layout (get_v for non-flash attn): kv is dim0 with nb0=elsize => contiguous across kv
        const size_t v_block_bytes = (size_t)task.v_n_new * v_nb0; // bytes across kv for a single (head, emb)
        const uint8_t* vsrc = static_cast<const uint8_t*>(task.v_data);
        for (int64_t h = 0; h < v_ne1; ++h) {
            for (int64_t e = 0; e < v_ne2; ++e) {
                const size_t tensor_off = (size_t)h * v_nb1 + (size_t)e * v_nb2 + (size_t)task.v_head * v_nb0;
                vfs.seekp((std::streamoff)tensor_off, std::ios::beg);
                vfs.write(reinterpret_cast<const char*>(vsrc), (std::streamsize)v_block_bytes);
                vsrc += v_block_bytes;
                if (vfs.fail()) {
                    printf("[KV-SSD] ✗ V delta write failed: layer %u, head %ld, emb %ld\n", task.layer_id, (long)h, (long)e);
                    return;
                }
            }
        }

        kfs.flush();
        vfs.flush();

        // free packed buffers
        if (task.k_data) free(task.k_data);
        if (task.v_data) free(task.v_data);
    }
    
    void execute_update_size(const kv_task& task) {}
    
    void execute_load(const kv_task& task) {
        reset_prefetch_status(task.layer_id);
        
        // Use tensor sizes directly
        size_t k_size = ggml_nbytes(task.k_tensor);
        size_t v_size = ggml_nbytes(task.v_tensor);
        
        void * k_copy = malloc(k_size);
        void * v_copy = malloc(v_size);

        if (!k_copy || !v_copy) {
            free(k_copy);
            free(v_copy);
            printf("[KV-SSD] ✗ Failed to allocate memory for layer %d\n", task.layer_id);
            return;
        }

        std::string kfile = cache_dir + "/layer_" + std::to_string(task.layer_id) + "_K.bin";
        std::string vfile = cache_dir + "/layer_" + std::to_string(task.layer_id) + "_V.bin";

        // K
        if (!std::filesystem::exists(kfile)) {
            printf("[KV-SSD] ⚠ Layer %d: K file does not exist: %s\n", task.layer_id, kfile.c_str());
            free(k_copy);
            free(v_copy);
            return;
        }
        // V
        if (!std::filesystem::exists(vfile)) {
            printf("[KV-SSD] ⚠ Layer %d: V file does not exist: %s\n", task.layer_id, vfile.c_str());
            free(k_copy);
            free(v_copy);
            return;
        }
        
        std::ifstream fk(kfile, std::ios::binary);
        std::ifstream fv(vfile, std::ios::binary);
        if (!fk.is_open() || !fv.is_open()) {
            printf("[KV-SSD] ✗ Failed to open K/V file for layer %d\n", task.layer_id);
            free(k_copy);
            free(v_copy);
            return;
        }
        
        try {
            std::error_code ec1, ec2;
            size_t k_file_size = std::filesystem::file_size(kfile, ec1);
            size_t v_file_size = std::filesystem::file_size(vfile, ec2);
            size_t k_read = std::min(k_size, k_file_size);
            size_t v_read = std::min(v_size, v_file_size);

            if (k_read == 0 && v_read == 0) {
                fk.close(); fv.close();
                free(k_copy);
                free(v_copy);
                return;
            }
            
            if (k_read > 0) {
                fk.read(static_cast<char*>(k_copy), (std::streamsize)k_read);
                if (fk.fail()) {
                    printf("[KV-SSD] ✗ Failed to read K data for layer %d (req=%zu, avail=%zu)\n", task.layer_id, k_size, k_file_size);
                    fk.close(); fv.close();
                    free(k_copy);
                    free(v_copy);
                    return;
                }
            }
            if (v_read > 0) {
                fv.read(static_cast<char*>(v_copy), (std::streamsize)v_read);
                if (fv.fail()) {
                    printf("[KV-SSD] ✗ Failed to read V data for layer %d (req=%zu, avail=%zu)\n", task.layer_id, v_size, v_file_size);
                    fk.close(); fv.close();
                free(k_copy);
                free(v_copy);
                return;
                }
            }
            
            // === 확인 코드 ===
            (void) verify_data_integrity(task, k_copy, v_copy, k_read, v_read);
            
            ggml_backend_tensor_set(task.k_tensor, k_copy, 0, k_read);
            ggml_backend_tensor_set(task.v_tensor, v_copy, 0, v_read);

            free(k_copy);
            free(v_copy);

            fk.close();
            fv.close();

        } catch (...) {
            printf("[KV-SSD] ✗ Exception during load for layer %d\n", task.layer_id);
            fk.close();
            fv.close();
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
        return offloader->save_layer(layer_id, k_tensor, v_tensor, 0, 0, 0, 0); // Placeholder for head/n_new
    }

bool llama_kv_offloader_load_layer(
    struct llama_kv_offloader* offloader,
    uint32_t layer_id,
    ggml_tensor* k_tensor,
    ggml_tensor* v_tensor) {
        if (!offloader) return false;
        return offloader->load_layer(layer_id, k_tensor, v_tensor);
    }

// C++ 버전: 벡터 참조 직접 반환

} // extern "C"

#ifdef __cplusplus
const std::vector<double>& llama_kv_offloader_get_load_times(struct llama_kv_offloader* offloader) {
    static std::vector<double> empty_vector;
    if (!offloader) return empty_vector;
    return offloader->load_times;
}
#endif

// =============================================================================
// GGML Backend Scheduler Callback Implementation (ADDED)
// =============================================================================

extern "C" {

    bool llama_kv_ggml_eval_callback(
        struct ggml_tensor * tensor, 
        bool ask, 
        void * user_data) {
        bool is_k = false;
        bool is_v = false;
        auto* cb_data = static_cast<llama_kv_callback_data*>(user_data);

        if (ask) {
        if (std::strncmp(tensor->name, "kv_sync", 7) == 0) {
            cb_data->is_sync = true;
            return true;
        } else if (std::strncmp(tensor->name, "k_cache_load", 12) == 0 || std::strncmp(tensor->name, "v_cache_load", 12) == 0) {
            cb_data->is_load = true;
            return true;
        } else if (std::strncmp(tensor->name, "k_cache_save", 12) == 0 || std::strncmp(tensor->name, "v_cache_save", 12) == 0) {
            cb_data->is_save = true;
            return true;
        } else {
            return false;
        }
    }
    if (cb_data->is_sync) {
        const char* dash_pos = std::strrchr(tensor->name, '-');
        int layer_id = std::atoi(dash_pos + 1);
        // printf("[KV-SSD] sync layer %d\n", layer_id);
        if (cb_data->offloader->is_first == 0) { 
            if (!cb_data->offloader->is_prefetch_complete(layer_id)) {
                llama_kv_offloader_wait_loads(cb_data->offloader);
            }
        } else if (layer_id == 31) {
            cb_data->offloader->is_first = 0;
        }
        cb_data->is_sync = false;
        return true;
    } else if (cb_data->is_load) {
        const char* dash_pos = std::strrchr(tensor->name, '-');
        int layer_id = std::atoi(dash_pos + 1);
        is_k = std::strncmp(tensor->name, "k_cache_load", 12) == 0;
        is_v = std::strncmp(tensor->name, "v_cache_load", 12) == 0;
        if (is_k) {
            cb_data->k_tensor = tensor;
            cb_data->k_cache_ready = true;
            cb_data->layer_id = layer_id;
        } else if (is_v) {
            cb_data->v_tensor = tensor;
            cb_data->v_cache_ready = true;
            cb_data->layer_id = layer_id;
        }
        if (cb_data->k_cache_ready && cb_data->v_cache_ready) {
            if (cb_data->offloader->get_pending_loads() > 31) {
                llama_kv_offloader_wait_loads(cb_data->offloader);
            }
            cb_data->offloader->load_layer(
                cb_data->layer_id,
                cb_data->k_tensor,
                cb_data->v_tensor
            );
            // printf("[KV-SSD] loading layer %d\n", layer_id);
            // llama_kv_offloader_wait_loads(cb_data->offloader);
            cb_data->k_tensor = nullptr;
            cb_data->v_tensor = nullptr;
            cb_data->k_cache_ready = cb_data->v_cache_ready = false;
        }
        cb_data->is_load = false;
        return true;
    } else if (cb_data->is_save) {
        // parse name: k_cache_save-<head>-<n_new>-<layer> or v_cache_save-<head>-<n_new>-<layer>
        const char* name = tensor->name;
        is_k = std::strncmp(name, "k_cache_save", 12) == 0;
        is_v = std::strncmp(name, "v_cache_save", 12) == 0;
        // layer id is the last dash-suffixed integer
        const char* last_dash = std::strrchr(name, '-');
        int layer_id = 0;
        if (last_dash) {
            layer_id = std::atoi(last_dash + 1);
        }
        // extract head and n_new from the part before last_dash
        int head = 0, n_new = 0;
        if (last_dash) {
            // copy prefix into buffer
            size_t prefix_len = (size_t)(last_dash - name);
            char buf[128];
            if (prefix_len >= sizeof(buf)) prefix_len = sizeof(buf) - 1;
            std::memcpy(buf, name, prefix_len);
            buf[prefix_len] = '\0';
            const char* p1 = std::strchr(buf, '-'); // after k_cache_save
            if (p1) {
                head = std::atoi(p1 + 1);
                const char* p2 = std::strchr(p1 + 1, '-');
                if (p2) n_new = std::atoi(p2 + 1);
            }
        }
        if (is_k) {
            cb_data->k_tensor = tensor;
            cb_data->k_cache_ready = true;
            cb_data->layer_id = layer_id;
            cb_data->k_head = head;
            cb_data->k_n_new = n_new;
        } else if (is_v) {
            cb_data->v_tensor = tensor;
            cb_data->v_cache_ready = true;
            cb_data->layer_id = layer_id;
            cb_data->v_head = head;
            cb_data->v_n_new = n_new;
        }
        if (cb_data->k_cache_ready && cb_data->v_cache_ready) {
            if (cb_data->offloader->get_pending_saves() > 31) {
                llama_kv_offloader_wait_saves(cb_data->offloader);
            }
            cb_data->offloader->save_layer(
                cb_data->layer_id,
                cb_data->k_tensor,
                cb_data->v_tensor,
                cb_data->k_head,
                cb_data->k_n_new,
                cb_data->v_head,
                cb_data->v_n_new
            );
            // llama_kv_offloader_wait_saves(cb_data->offloader);
            // printf("[KV-SSD] saving layer %d\n", layer_id);
            cb_data->k_tensor = nullptr;
            cb_data->v_tensor = nullptr;
            cb_data->k_cache_ready = cb_data->v_cache_ready = false;
        }
        cb_data->is_save = false;
        return true;
    }
    }
    


} // extern "C"
