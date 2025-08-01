# KV Cache SSD Offloading 시스템 설계 (Compression & 확장성 고려)

## 1. 전체 아키텍처 개요

```
Main Thread (Inference)           Worker Thread Pool (4-8개)
┌─────────────────────┐          ┌─────────────────────────┐
│ llama_context       │          │ Thread 0: Save Worker  │
│ - decode()          │─submit──▶│ Thread 1: Load Worker  │
│ - kv_get_data()     │          │ Thread 2: Save Worker  │
│ - immediate save    │          │ Thread 3: Load Worker  │
└─────────────────────┘          └─────────────────────────┘
         │                                    │
         │ full copy → compression            │
         ▼                                    ▼
┌─────────────────────┐          ┌─────────────────────────┐
│ Memory Slots (N개)  │◀────────▶│    Thread-Safe Queue    │
│ - Compressed KV     │          │ - save_queue            │
│ - Decompressed KV   │          │ - load_queue            │
└─────────────────────┘          └─────────────────────────┘
                                           │
                                           ▼
                                 ┌─────────────────────────┐
                                 │    SSD Storage          │
                                 │ compressed_layer_X.bin  │
                                 └─────────────────────────┘
```

## 2. 핵심 설계 철학

### 2.1 Memory Slot의 역할 변화
- **현재**: 실제 KV cache의 완전한 복사본 저장
- **미래**: Compression/Decompression을 위한 중간 버퍼 역할
- **확장성**: 2-layer → N-layer 확장 가능한 구조

### 2.2 Compression 적용 전략
```cpp
struct memory_slot {
    uint32_t layer_id;                        // 레이어 ID
    bool is_valid;                            // 유효한 데이터 여부
    bool is_loading;                          // 비동기 로드 진행 중
    
    // === 현재: 완전 복사 방식 ===
    void* k_data;                             // K 텐서 원본 데이터
    void* v_data;                             // V 텐서 원본 데이터
    size_t data_size;                         // 할당된 메모리 크기 (원본)
    
    // === 미래: Compression 추가 ===
    void* compressed_k_data;                  // 압축된 K 텐서 (향후)
    void* compressed_v_data;                  // 압축된 V 텐서 (향후)
    size_t compressed_size;                   // 압축된 크기 (향후)
    compression_method_t compression_type;    // 압축 방식 (향후)
    
    size_t seq_len;                           // 현재 시퀀스 길이
    std::chrono::steady_clock::time_point last_access;  // LRU용
    std::mutex slot_mutex;                    // 슬롯 보호 뮤텍스
};
```

### 2.3 확장 가능한 슬롯 관리
```cpp
class llama_kv_offloader {
private:
    // === 현재: 2-layer 고정 ===
    std::array<memory_slot, 2> memory_slots;
    
    // === 미래: N-layer 확장 ===
    // std::vector<memory_slot> memory_slots;  // 동적 크기
    // uint32_t max_layers_in_memory;          // 설정 가능한 크기
    
    std::string cache_dir;
    uint32_t num_worker_threads;
    
    // Thread-safe 작업 큐들
    std::queue<kv_task> save_queue;
    std::queue<kv_task> load_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    
    // Worker threads
    std::vector<std::thread> worker_threads;
    std::atomic<bool> shutdown_flag;
    
    // 완료 추적
    std::atomic<uint32_t> pending_save_ops;
    std::atomic<uint32_t> pending_load_ops;
    std::mutex completion_mutex;
    std::condition_variable completion_cv;
    
public:
    // === 현재 API (2-layer 전용) ===
    bool save_layer_immediately(uint32_t layer_id);
    bool process_and_save_layer(uint32_t layer_id);
    
    // === 미래 API (N-layer 확장) ===
    // bool configure_memory_slots(uint32_t num_slots);
    // bool set_compression_method(compression_method_t method);
    // bool enable_adaptive_compression(bool enable);
};
```

## 3. 데이터 복사 전략의 정당성

### 3.1 왜 완전 복사가 필요한가?

#### A. Compression 요구사항
```cpp
// 현재: 원본 KV 데이터 완전 복사
bool save_layer_with_full_copy(uint32_t layer_id) {
    // 1. llama_context에서 완전 복사
    size_t layer_size = llama_kv_layer_get_size(ctx, layer_id);
    slot->k_data = malloc(layer_size / 2);
    slot->v_data = malloc(layer_size / 2);
    llama_kv_layer_get_data(ctx, layer_id, slot->k_data, slot->v_data);
    
    // 2. 복사본을 SSD에 저장
    save_to_ssd(layer_id, slot->k_data, slot->v_data);
    
    return true;
}

// 미래: Compression 적용
bool save_layer_with_compression(uint32_t layer_id) {
    // 1. llama_context에서 완전 복사 (동일)
    llama_kv_layer_get_data(ctx, layer_id, slot->k_data, slot->v_data);
    
    // 2. 복사본을 압축
    slot->compressed_size = compress_kv_data(
        slot->k_data, slot->v_data, slot->data_size,
        slot->compressed_k_data, slot->compressed_v_data,
        compression_type
    );
    
    // 3. 압축된 데이터를 SSD에 저장
    save_compressed_to_ssd(layer_id, slot->compressed_k_data, 
                          slot->compressed_v_data, slot->compressed_size);
    
    return true;
}
```

#### B. 독립성 보장
```cpp
// llama_context의 KV cache는 계속 변경됨
while (inference_continues) {
    // Layer 0 처리 중
    process_layer(0);
    
    // Layer 0을 메모리 슬롯에 복사 (독립적 보관)
    copy_to_memory_slot(0);
    
    // Layer 1로 이동 - Layer 0의 KV cache는 변경됨
    process_layer(1);
    
    // 하지만 메모리 슬롯의 Layer 0 데이터는 안전하게 보존됨
    save_slot_to_ssd(0);  // 원본 데이터 그대로 저장
}
```

### 3.2 메모리 사용량 최적화

#### 현재 (2-layer): 제한적 메모리 사용
```cpp
// 메모리 사용량 = 2 * layer_size
memory_usage = 2 * (k_tensor_size + v_tensor_size)
             = 2 * layer_kv_size
```

#### 미래 (N-layer + Compression): 유연한 메모리 관리
```cpp
// 압축 시 메모리 절약
compressed_memory_usage = N * layer_kv_size * compression_ratio
                        = N * layer_kv_size * 0.3  // 예: 70% 압축

// 동적 슬롯 수 조정
if (available_memory > threshold) {
    increase_memory_slots();  // 더 많은 레이어 캐싱
} else {
    decrease_memory_slots();  // 메모리 절약
}
```

## 4. 확장성 로드맵

### 4.1 Phase 1: 현재 (2-layer 고정)
- ✅ 2개 메모리 슬롯 고정
- ✅ 완전 복사 방식
- ✅ 즉시 저장 전략
- ✅ 비동기 I/O

### 4.2 Phase 2: Compression 추가
```cpp
// 압축 방식 선택
enum compression_method_t {
    NONE,           // 압축 없음 (현재)
    LZ4,            // 빠른 압축
    ZSTD,           // 균형잡힌 압축
    QUANTIZATION,   // KV-specific 양자화
    CUSTOM          // 커스텀 압축
};

// 압축 적용
bool apply_compression(memory_slot* slot) {
    switch (slot->compression_type) {
        case LZ4:
            return compress_with_lz4(slot);
        case ZSTD:
            return compress_with_zstd(slot);
        case QUANTIZATION:
            return quantize_kv_data(slot);
        default:
            return true;  // 압축 없음
    }
}
```

### 4.3 Phase 3: N-layer 확장
```cpp
class llama_kv_offloader {
private:
    // 동적 슬롯 관리
    std::vector<memory_slot> memory_slots;
    uint32_t max_layers_in_memory;
    
    // LRU 정책 고도화
    std::map<uint32_t, std::chrono::steady_clock::time_point> access_history;
    
public:
    // 슬롯 수 동적 조정
    bool resize_memory_slots(uint32_t new_size) {
        if (new_size < 2) return false;  // 최소 2개
        
        memory_slots.resize(new_size);
        max_layers_in_memory = new_size;
        return true;
    }
    
    // 적응형 슬롯 관리
    bool enable_adaptive_slots(bool enable) {
        // 메모리 사용량에 따라 자동 조정
        return configure_adaptive_management(enable);
    }
};
```

## 5. 설계 규칙 업데이트

### 5.1 메모리 슬롯 설계 원칙
1. **완전 복사**: llama_context에서 독립적인 복사본 생성
2. **압축 준비**: 향후 compression 적용을 위한 구조
3. **확장성**: 2-layer → N-layer 확장 가능한 아키텍처
4. **독립성**: 원본 KV cache와 완전히 분리된 관리

### 5.2 현재 vs 미래 전략

#### 현재 구현 (Phase 1)
- 2개 고정 슬롯
- 원본 데이터 완전 복사
- 즉시 저장 (no dirty flag)
- 단순한 LRU 정책

#### 미래 확장 (Phase 2-3)
- N개 동적 슬롯
- 압축된 데이터 저장
- 적응형 메모리 관리
- 고급 eviction 정책

### 5.3 API 호환성 보장
```cpp
// 현재 API는 미래에도 유지
bool llama_kv_offloader_save_layer_immediately(
    struct llama_kv_offloader* offloader,
    uint32_t layer_id
);

// 확장 API는 추가로 제공
bool llama_kv_offloader_save_layer_compressed(
    struct llama_kv_offloader* offloader,
    uint32_t layer_id,
    compression_method_t method
);

bool llama_kv_offloader_configure_slots(
    struct llama_kv_offloader* offloader,
    uint32_t num_slots
);
```

## 6. 성능 고려사항

### 6.1 메모리 복사 오버헤드
- **현재**: 필요한 오버헤드 (compression 준비)
- **최적화**: SIMD 명령어 활용한 고속 복사
- **미래**: 압축으로 인한 메모리 절약이 복사 비용 상쇄

### 6.2 확장성 vs 성능
```cpp
// 슬롯 수에 따른 성능 트레이드오프
if (num_slots == 2) {
    // 최소 메모리, 최대 I/O
    performance_profile = {.memory_low = true, .io_high = true};
} else if (num_slots > 8) {
    // 최대 메모리, 최소 I/O
    performance_profile = {.memory_high = true, .io_low = true};
} else {
    // 균형잡힌 성능
    performance_profile = {.memory_medium = true, .io_medium = true};
}
```

## 7. 마이그레이션 전략

### 7.1 단계별 업그레이드
1. **Phase 1**: 현재 구현 (2-layer, 복사, 즉시저장)
2. **Phase 2**: Compression 추가 (기존 API 유지)
3. **Phase 3**: N-layer 확장 (새로운 API 추가)

### 7.2 호환성 보장
- 기존 API는 항상 유지
- 새로운 기능은 추가 API로 제공
- 설정을 통한 기능 활성화/비활성화

이 설계는 현재의 단순함을 유지하면서도 미래의 고급 기능을 자연스럽게 확장할 수 있는 견고한 기반을 제공합니다.
