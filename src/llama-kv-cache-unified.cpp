#include "llama-kv-cache-unified.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

//
// llama_kv_cache_unified - 통합 KV Cache 관리 클래스
//
// 이 클래스는 Transformer 모델의 Key-Value cache를 효율적으로 관리합니다.
// 주요 기능:
// 1. 다중 시퀀스 지원 (병렬 생성)
// 2. 메모리 최적화 (defragmentation, shift)
// 3. GPU/CPU 백엔드 지원
// 4. Sliding Window Attention (SWA) 지원
// 5. 상태 저장/로드 (세션 복원)
//

/**
 * llama_kv_cache_unified 생성자
 * 
 * KV cache의 전체 구조를 초기화하고 메모리를 할당합니다.
 * 
 * @param model: 모델 정보 (하이퍼파라미터, 레이어 구성 등)
 * @param filter: 레이어 필터 함수 (특정 레이어만 캐시할 때 사용)
 * @param type_k: Key 텐서의 데이터 타입 (f16, q8_0 등)
 * @param type_v: Value 텐서의 데이터 타입
 * @param v_trans: Value 텐서 전치 여부 (flash attention 사용 시 false)
 * @param offload: GPU 오프로딩 사용 여부
 * @param kv_size: KV cache 총 크기 (토큰 수)
 * @param n_seq_max: 최대 동시 시퀀스 수
 * @param n_pad: 패딩 크기 (메모리 정렬용)
 * @param n_swa: Sliding Window 크기
 * @param swa_type: SWA 타입 (NONE, STANDARD, CHUNKED)
 */
llama_kv_cache_unified::llama_kv_cache_unified(
        const llama_model &  model,
          layer_filter_cb && filter,
                ggml_type    type_k,
                ggml_type    type_v,
                     bool    v_trans,
                     bool    offload,
                 uint32_t    kv_size,
                 uint32_t    n_seq_max,
                 uint32_t    n_pad,
                 uint32_t    n_swa,
           llama_swa_type    swa_type) :
    model(model), hparams(model.hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type) {

    // kv_size는 n_pad의 배수여야 함 (메모리 정렬 요구사항)
    GGML_ASSERT(kv_size % n_pad == 0);

    // ==================== 백엔드별 컨텍스트 생성 ====================
    // 다양한 백엔드(CPU, GPU)에 대해 별도의 ggml 컨텍스트를 생성
    // 이렇게 하면 텐서들을 적절한 디바이스에 배치할 수 있음
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // 새로운 백엔드 타입이면 새 컨텍스트 생성
            ggml_init_params params = {
                // 메모리 크기: 각 레이어마다 텐서 메타데이터 저장 공간
                /*.mem_size   =*/ size_t(2u*hparams.n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,  // NULL이면 내부적으로 할당
                /*.no_alloc   =*/ true,  // 텐서 생성 시 실제 데이터는 나중에 할당
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);  // 나중에 해제하기 위해 저장

            return ctx;
        }

        return it->second;
    };

    // ==================== KV Cache 구조 초기화 ====================

    head = 0;  // 다음 할당할 위치를 가리키는 포인터

    // cells: KV cache의 메타데이터 저장 (위치, 시퀀스 ID, shift 정보 등)
    // 실제 텐서 데이터는 layers에 저장됨
    cells.resize(kv_size);

    // ==================== 레이어별 KV 텐서 생성 ====================

    for (uint32_t il = 0; il < hparams.n_layer; il++) {
        // 레이어 필터가 있고, 이 레이어를 건너뛰라고 하면 스킵
        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, il);
            continue;
        }

        // GQA (Grouped Query Attention) 차원 계산
        // Key와 Value의 차원이 다를 수 있음 (메모리 절약)
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);  // Key 임베딩 차원
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);  // Value 임베딩 차원

        const char * dev_name = "CPU";

        // ==================== 백엔드 선택 ====================
        
        // 기본적으로 CPU 백엔드 사용
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            // GPU 오프로딩이 활성화되면 해당 레이어의 디바이스 확인
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        // 해당 백엔드의 컨텍스트 가져오기 (없으면 생성)
        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        // ==================== Key, Value 텐서 생성 ====================

        ggml_tensor * k;
        ggml_tensor * v;

        // Key 텐서: [n_embd_k_gqa, kv_size] 형태
        // - n_embd_k_gqa: Key의 임베딩 차원
        // - kv_size: 저장할 수 있는 최대 토큰 수
        k = ggml_new_tensor_2d(ctx, type_k, n_embd_k_gqa, kv_size);
        
        // Value 텐서: [n_embd_v_gqa, kv_size] 형태
        // v_trans가 true면 실제로는 전치되어 저장됨
        v = ggml_new_tensor_2d(ctx, type_v, n_embd_v_gqa, kv_size);

        // 디버깅을 위한 텐서 이름 설정
        ggml_format_name(k, "cache_k_l%d", il);  // "cache_k_l0", "cache_k_l1", ...
        ggml_format_name(v, "cache_v_l%d", il);  // "cache_v_l0", "cache_v_l1", ...

        // 레이어 ID → 내부 인덱스 매핑 저장
        // 모든 레이어가 KV cache를 사용하는 것은 아니므로 매핑 필요
        map_layer_ids[il] = layers.size();
        
        // 레이어 정보 저장
        layers.push_back({ il, k, v });
    }

    // ==================== 실제 메모리 할당 ====================
    
    // 지금까지는 텐서 구조만 만들었고, 실제 메모리는 아직 할당 안됨
    // 이제 백엔드별로 실제 메모리를 할당하고 초기화
    for (auto it : ctx_map) {
        auto * buft = it.first;  // 백엔드 버퍼 타입
        auto * ctx  = it.second; // ggml 컨텍스트

        // 해당 컨텍스트의 모든 텐서에 대해 실제 메모리 할당
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        // 메모리 사용량 로그 출력
        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, 
                       ggml_backend_buffer_name(buf), 
                       ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        // 할당된 메모리를 0으로 초기화 (패딩 영역의 NaN 방지)
        ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);  // 나중에 해제하기 위해 저장
    }

    // ==================== 메모리 사용량 요약 ====================

    {
        const size_t memory_size_k = size_k_bytes();  // 모든 Key 텐서의 총 크기
        const size_t memory_size_v = size_v_bytes();  // 모든 Value 텐서의 총 크기

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), 
                kv_size,                    // 총 셀 수
                (int) layers.size(),        // 실제 사용하는 레이어 수
                n_seq_max,                  // 최대 시퀀스 수
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // ==================== 디버그 모드 설정 ====================
    
    // 환경 변수 LLAMA_KV_CACHE_DEBUG로 디버그 레벨 조절 가능
    // 0: 디버그 없음, 1: 기본 정보, 2: 상세 정보, 3: 매우 상세
    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

/**
 * KV cache 전체 초기화
 * 
 * @param data: true면 실제 텐서 데이터도 0으로 초기화, false면 메타데이터만 초기화
 * 
 * 사용 시점:
 * - 새로운 세션 시작 시
 * - 컨텍스트 완전 리셋 시
 * - 오류 복구 시
 */
void llama_kv_cache_unified::clear(bool data) {
    // 모든 셀의 메타데이터 초기화 (위치, 시퀀스 ID, shift 정보 등)
    cells.reset();

    // head를 처음으로 되돌림 (다음 할당이 0번째 셀부터 시작)
    head = 0;

    if (data) {
        // 실제 텐서 데이터도 0으로 초기화 (메모리 내용 삭제)
        // 보안상 중요한 정보나 디버깅 시 이전 데이터의 간섭을 방지
        for (auto & buf : bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

/**
 * 특정 시퀀스의 지정된 위치 범위 제거
 * 
 * @param seq_id: 제거할 시퀀스 ID (-1이면 모든 시퀀스)
 * @param p0: 시작 위치 (포함, -1이면 처음부터)
 * @param p1: 끝 위치 (제외, -1이면 끝까지)
 * @return: 성공 여부
 * 
 * 사용 시점:
 * - 토큰 생성 실패 시 롤백
 * - 메모리 압박 시 오래된 데이터 정리
 * - 시퀀스 완료 시 정리
 * - SSD offloading 시 메모리 확보용
 */
bool llama_kv_cache_unified::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = cells.size();  // 해제된 셀 중 가장 앞쪽 위치 추적

    // 범위 정규화
    if (p0 < 0) {
        p0 = 0;  // 처음부터
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();  // 끝까지
    }

    if (seq_id >= 0) {
        // ==================== 특정 시퀀스만 제거 ====================
        for (uint32_t i = 0; i < cells.size(); ++i) {
            // 지정된 위치 범위에 속하지 않으면 건너뛰기
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            // 해당 셀이 지정된 시퀀스를 포함하고 있는지 확인
            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                // 시퀀스 제거 후 셀이 완전히 비었으면 head 업데이트 후보
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    } else {
        // ==================== 모든 시퀀스 제거 (위치 범위 내) ====================
        for (uint32_t i = 0; i < cells.size(); ++i) {
            // 지정된 위치 범위에 속하지 않으면 건너뛰기
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            // 해당 셀의 모든 시퀀스 제거
            cells.rm(i);

            // head 업데이트 후보
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // ==================== Head 포인터 최적화 ====================
    // 해제된 셀이 있고, 현재 head보다 앞쪽에 있으면 head를 그곳으로 이동
    // 이렇게 하면 다음 할당 시 앞쪽 빈 공간부터 사용하게 됨 (메모리 효율성)
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }

    return true;
}

/**
 * 시퀀스 복사 (한 시퀀스의 KV cache를 다른 시퀀스 ID로 복사)
 * 
 * @param seq_id_src: 원본 시퀀스 ID
 * @param seq_id_dst: 대상 시퀀스 ID
 * @param p0: 시작 위치 (포함, -1이면 처음부터)
 * @param p1: 끝 위치 (제외, -1이면 끝까지)
 * 
 * 사용 시점:
 * - Beam search에서 후보 분기 생성
 * - 병렬 생성에서 공통 프롬프트 공유
 * - 대화 컨텍스트 복사 (새 대화 시작 시)
 * 
 * 주의: 실제 텐서 데이터는 복사하지 않고, 메타데이터만 공유
 * 같은 물리적 KV cache를 여러 시퀀스가 참조하게 됨
 */
void llama_kv_cache_unified::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // 같은 시퀀스면 아무것도 할 필요 없음
    if (seq_id_src == seq_id_dst) {
        return;
    }

    // 범위 정규화
    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // ==================== 시퀀스 복사 처리 ====================
    for (uint32_t i = 0; i < cells.size(); ++i) {
        // 지정된 위치 범위에 속하지 않으면 건너뛰기
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        // 원본 시퀀스를 포함하는 셀이면 대상 시퀀스도 추가
        if (cells.seq_has(i, seq_id_src)) {
            cells.seq_add(i, seq_id_dst);
        }
    }
}

/**
 * 특정 시퀀스만 유지하고 나머지 모두 제거
 * 
 * @param seq_id: 유지할 시퀀스 ID
 * 
 * 사용 시점:
 * - 병렬 생성에서 하나의 결과만 선택할 때
 * - 메모리 압박 시 가장 중요한 시퀀스만 유지
 * - 세션 정리 시
 */
void llama_kv_cache_unified::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = cells.size();

    // 모든 셀을 순회하면서 지정된 시퀀스만 유지
    for (uint32_t i = 0; i < cells.size(); ++i) {
        // seq_keep: 지정된 시퀀스만 유지하고 나머지 제거
        // 반환값이 true면 셀이 완전히 비워졌다는 의미
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // Head 포인터 최적화
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

/**
 * 시퀀스의 위치를 이동 (shift)
 * 
 * @param seq_id: 이동할 시퀀스 ID
 * @param p0: 시작 위치 (포함)
 * @param p1: 끝 위치 (제외)
 * @param shift: 이동할 양 (양수면 앞으로, 음수면 뒤로)
 * 
 * 사용 시점:
 * - RoPE (Rotary Position Embedding) 위치 조정
 * - 컨텍스트 윈도우 슬라이딩
 * - 토큰 삽입/삭제 시 위치 재정렬
 */
void llama_kv_cache_unified::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // shift가 0이면 아무것도 할 필요 없음
    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    // 범위 정규화
    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // 범위가 비어있으면 조기 종료 (성능 최적화)
    if (p0 == p1) {
        return;
    }

    // ==================== 위치 이동 처리 ====================
    for (uint32_t i = 0; i < cells.size(); ++i) {
        // 지정된 위치 범위에 속하지 않으면 건너뛰기
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        // 해당 시퀀스를 포함하는 셀이면 위치 이동
        if (cells.seq_has(i, seq_id)) {
            // pos_add가 true를 반환하면 셀이 비워졌다는 의미
            // (위치 이동 결과 유효하지 않은 위치가 되었을 때)
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // Head 포인터 최적화
    // 새로 비워진 셀이 있으면 그곳부터 검색, 없으면 처음부터 검색
    head = new_head != cells.size() ? new_head : 0;
}

/**
 * 시퀀스의 위치를 나누기 (압축)
 * 
 * @param seq_id: 대상 시퀀스 ID
 * @param p0: 시작 위치 (포함)
 * @param p1: 끝 위치 (제외)
 * @param d: 나눌 값 (d=2이면 위치가 절반으로)
 * 
 * 사용 시점:
 * - 긴 시퀀스의 위치 압축
 * - 메모리 절약을 위한 위치 정규화
 * - 특정 알고리즘에서 위치 스케일링
 */
void llama_kv_cache_unified::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // d가 1이면 아무것도 할 필요 없음
    if (d == 1) {
        return;
    }

    // 범위 정규화
    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // 범위가 비어있으면 조기 종료
    if (p0 == p1) {
        return;
    }

    // ==================== 위치 압축 처리 ====================
    for (uint32_t i = 0; i < cells.size(); ++i) {
        // 지정된 위치 범위에 속하지 않으면 건너뛰기
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        // 해당 시퀀스를 포함하는 셀이면 위치를 d로 나누기
        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

/**
 * 특정 시퀀스의 최소 위치 반환
 * 
 * @param seq_id: 조회할 시퀀스 ID
 * @return: 해당 시퀀스의 최소 위치 (-1이면 시퀀스가 없음)
 * 
 * 사용 용도:
 * - 시퀀스의 시작점 확인
 * - Sliding Window 계산
 * - 메모리 정리 범위 결정
 */
llama_pos llama_kv_cache_unified::seq_pos_min(llama_seq_id seq_id) const {
    return cells.seq_pos_min(seq_id);
}

/**
 * 특정 시퀀스의 최대 위치 반환
 * 
 * @param seq_id: 조회할 시퀀스 ID  
 * @return: 해당 시퀀스의 최대 위치 (-1이면 시퀀스가 없음)
 * 
 * 사용 용도:
 * - 시퀀스의 끝점 확인
 * - 다음 토큰 위치 계산
 * - Attention mask 생성
 */
llama_pos llama_kv_cache_unified::seq_pos_max(llama_seq_id seq_id) const {
    return cells.seq_pos_max(seq_id);
}

/**
 * 배치 초기화 - KV cache에 배치를 처리할 수 있는지 확인하고 준비
 * 
 * 이 함수는 KV cache 관리의 핵심으로, 다음과 같은 과정을 거칩니다:
 * 1. 입력 배치를 내부 형식으로 변환
 * 2. micro-batch로 분할
 * 3. 각 micro-batch가 KV cache에 들어갈 수 있는지 확인
 * 4. 메모리 상태 객체 생성 및 반환
 * 
 * @param batch: 처리할 입력 배치
 * @param n_ubatch: micro-batch 크기
 * @param embd_all: 모든 토큰의 임베딩 출력 여부
 * @return: 메모리 상태 객체 (실패 시 FAILED_PREPARE 상태)
 * 
 * SSD Offloading 관점에서 중요한 이유:
 * - 메모리 부족 감지 지점
 * - SSD에서 필요한 데이터 로드 시작점
 * - 불필요한 데이터 SSD 이동 결정 지점
 */
llama_memory_state_ptr llama_kv_cache_unified::init_batch(
            const llama_batch & batch,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);  // 현재 구현에서는 사용하지 않음

    do {
        // ==================== 배치 형식 변환 ====================
        
        // llama_sbatch: 내부 최적화된 배치 형식
        // - 메모리 정렬 최적화
        // - 시퀀스별 그룹화
        // - 출력 인덱스 매핑
        auto sbatch = llama_sbatch(batch, hparams.n_embd, true);

        // ==================== Micro-batch 분할 ====================

        std::vector<llama_ubatch> ubatches;
        
        // 큰 배치를 작은 micro-batch들로 나누기
        // 이유:
        // 1. GPU 메모리 제한 대응
        // 2. 메모리 사용량 조절
        // 3. 실패 시 부분 복구 가능
        // 4. SSD offloading 시 세밀한 제어
        while (sbatch.n_tokens > 0) {
            ubatches.push_back(sbatch.split_simple(n_ubatch));
        }

        // ==================== 배치 배치 계획 수립 ====================
        
        // 각 micro-batch가 KV cache에 들어갈 위치를 미리 계산
        // 실제로 데이터를 넣지는 않고, 들어갈 수 있는지만 확인
        auto heads = prepare(ubatches);
        
        if (heads.empty()) {
            // 배치할 공간이 없음 - 메모리 압박 상황
            // SSD offloading 시점:
            // - 여기서 메모리 부족을 감지
            // - 오래된 KV cache를 SSD로 이동 시도
            // - 필요한 KV cache를 SSD에서 로드 시도
            break;
        }

        // ==================== 성공: 메모리 상태 객체 생성 ====================

        return std::make_unique<llama_kv_cache_unified_state>(
                this,                    // KV cache 참조
                std::move(sbatch),       // 변환된 배치
                std::move(heads),        // 각 micro-batch의 head 위치
                std::move(ubatches));    // micro-batch 목록
                
    } while (false);

    // ==================== 실패: 에러 상태 반환 ====================

    return std::make_unique<llama_kv_cache_unified_state>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_state_ptr llama_kv_cache_unified::init_full() {
    return std::make_unique<llama_kv_cache_unified_state>(this);
}

llama_memory_state_ptr llama_kv_cache_unified::init_update(llama_context * lctx, bool optimize) {
    bool do_shift = get_has_shift();

    defrag_info dinfo;

    // see if we need to defrag
    {
        bool do_defrag = optimize;

        const auto thold = lctx->get_cparams().defrag_thold;

        if (!do_defrag && thold > 0.0f) {
            const auto n_kv = cells.used_max_p1();

            // - do not defrag small contexts (i.e. < 2048 tokens)
            // - count the padding towards the number of used tokens
            const float fragmentation = n_kv >= 2048 ? std::max(0.0f, 1.0f - (float(cells.get_used() + n_pad)/n_kv)) : 0.0f;

            if (fragmentation > thold) {
                LLAMA_LOG_DEBUG("%s: fragmentation: %.2f - requesting defrag\n", __func__, fragmentation);

                do_defrag = true;
            }
        }

        if (do_defrag) {
            dinfo = defrag_prepare(lctx->graph_max_nodes());
        }
    }

    return std::make_unique<llama_kv_cache_unified_state>(this, lctx, do_shift, std::move(dinfo));
}

/**
 * Micro-batch들이 KV cache에 배치될 수 있는지 사전 검증
 * 
 * 이 함수는 실제로 KV cache를 수정하지 않고, 각 micro-batch가 들어갈 위치만 계산합니다.
 * 가상으로 배치해보고 모든 micro-batch가 들어갈 수 있으면 위치 목록을 반환합니다.
 * 
 * 동작 원리:
 * 1. 각 micro-batch에 대해 find_slot()으로 위치 찾기
 * 2. 임시로 해당 위치에 배치 (메타데이터만)
 * 3. 다음 micro-batch 위치 찾기
 * 4. 모든 micro-batch 처리 후 원래 상태로 복원
 * 5. 성공하면 위치 목록 반환, 실패하면 빈 목록 반환
 * 
 * @param ubatches: 배치할 micro-batch 목록
 * @return: 각 micro-batch의 head 위치 목록 (실패 시 빈 목록)
 * 
 * SSD Offloading 핵심 지점:
 * - 메모리 부족 감지: find_slot()이 실패하는 지점
 * - 이 시점에서 SSD offloading 정책 결정
 * - 오래된 KV cache를 SSD로 이동
 * - 필요한 KV cache를 SSD에서 로드
 */
llama_kv_cache_unified::ubatch_heads llama_kv_cache_unified::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache_unified::ubatch_heads res;  // 결과: 각 micro-batch의 위치

    // ==================== 상태 백업 구조체 ====================

    struct state {
        uint32_t head_old; // 배치 전 head 위치
        uint32_t head_new; // 배치 후 head 위치

        llama_kv_cells_unified cells; // 배치 전 셀 상태 백업
    };

    // 각 micro-batch 배치 시점의 상태를 백업 (나중에 복원용)
    std::vector<state> states;

    bool success = true;

    // ==================== 가상 배치 시뮬레이션 ====================

    for (const auto & ubatch : ubatches) {
        // 현재 상태에서 이 micro-batch가 들어갈 위치 찾기
        // 아직 실제로 배치하지는 않음 (read-only 검색)
        const int32_t head_new = find_slot(ubatch);
        
        if (head_new < 0) {
            // 배치할 공간을 찾지 못함 = 메모리 부족
            // SSD Offloading 필요 시점:
            // 1. 오래된 시퀀스들을 SSD로 이동
            // 2. 현재 필요한 데이터를 SSD에서 로드
            // 3. 재시도 또는 실패 반환
            success = false;
            break;
        }

        // 찾은 위치를 결과에 추가
        res.push_back(head_new);

        // ==================== 상태 백업 및 임시 배치 ====================
        
        // 현재 상태를 백업 스택에 저장
        states.push_back({
            head,                           // 현재 head 위치
            (uint32_t) head_new,           // 찾은 새 head 위치
            cells.cp(head_new, ubatch.n_tokens)  // 영향받을 셀들의 상태 백업
        });

        // 임시로 이 micro-batch를 배치 (다음 micro-batch 위치 찾기 위해)
        // 실제 텐서 데이터는 건드리지 않고 메타데이터만 업데이트
        apply_ubatch(head_new, ubatch);
    }

    // ==================== 상태 복원 ====================
    
    // 배치 시뮬레이션이 완료되었으므로 원래 상태로 복원
    // 역순으로 복원해야 정확함 (스택 방식)
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        // 백업해둔 셀 상태로 복원
        cells.set(it->head_new, it->cells);
        
        // head 위치도 원래대로 복원
        head = it->head_old;
    }

    if (!success) {
        // 배치 실패: 빈 목록 반환
        // 호출자는 이를 보고 메모리 압박 상황으로 판단
        return {};
    }

    return res;  // 성공: 각 micro-batch의 위치 목록 반환
}

bool llama_kv_cache_unified::update(llama_context * lctx, bool do_shift, const defrag_info & dinfo) {
    bool updated = false;

    auto * sched = lctx->get_sched();

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * gf = lctx->graph_init();

            auto res = build_graph_shift(lctx->get_cparams(), lctx->get_ctx_compute(), gf);
            if (!res) {
                LLAMA_LOG_ERROR("%s: failed to build graph for K-shift\n", __func__);
                return updated;
            }

            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        cells.reset_shift();
    }

    if (!dinfo.empty()) {
        LLAMA_LOG_DEBUG("%s: defragmenting KV cache\n", __func__);

        // apply moves:
        {
            const auto n_kv = dinfo.ids.size();

            for (uint32_t i = 0; i < n_kv; ++i) {
                assert(dinfo.ids[i] <= n_kv);

                if (dinfo.ids[i] == n_kv || dinfo.ids[i] == i) {
                    continue;
                }

                cells.mv(i, dinfo.ids[i]);
            }

            // reset the head so we can find the first free slot during the next ubatch
            head = 0;
        }

        ggml_backend_sched_reset(sched);

        auto * gf = lctx->graph_init();

        auto res = build_graph_defrag(lctx->get_cparams(), lctx->get_ctx_compute(), gf, dinfo);
        if (!res) {
            LLAMA_LOG_ERROR("%s: failed to build graph for defrag\n", __func__);
            return updated;
        }

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate compute graph for defrag\n", __func__);
            return updated;
        }

        res->set_inputs(nullptr);

        if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: failed to compute defrag\n", __func__);
            return updated;
        }

        updated = true;
    }

    return updated;
}

/**
 * Micro-batch가 들어갈 적절한 위치 찾기
 * 
 * 이 함수는 KV cache 메모리 관리의 핵심으로, 다음 조건을 만족하는 위치를 찾습니다:
 * 1. 연속된 n_tokens 개의 셀이 비어있거나 재사용 가능
 * 2. Sliding Window Attention (SWA) 제약 조건 만족
 * 3. 시퀀스 충돌 방지
 * 
 * 검색 전략:
 * - 현재 head부터 시작하여 순환 검색
 * - 사용하지 않는 셀들이 많으면 처음부터 검색 (조각화 방지)
 * - First-fit 알고리즘 사용
 * 
 * @param ubatch: 배치할 micro-batch
 * @return: 배치할 시작 위치 (실패 시 -1)
 * 
 * SSD Offloading 최적화 지점:
 * - 메모리 부족 감지 시점
 * - LRU/LFU 정책으로 victim 선택
 * - 백그라운드 SSD 이동 트리거
 * - 필요 데이터 prefetch 시작점
 */
int32_t llama_kv_cache_unified::find_slot(const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    uint32_t head_cur = this->head;

    // ==================== 검색 시작점 최적화 ====================
    
    // 현재 head 앞쪽에 사용하지 않는 셀이 많으면 처음부터 검색
    // 이유: 메모리 조각화 방지 및 지역성 향상
    // 경험적 임계값: 필요한 토큰 수의 2배 + 현재 사용량
    if (head_cur > cells.get_used() + 2*ubatch.n_tokens) {
        head_cur = 0;
    }

    // ==================== 기본 검증 ====================

    if (n_tokens > cells.size()) {
        LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
        return -1;
    }

    // ==================== 디버그 정보 출력 ====================

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n", 
                __func__, 
                cells.used_max_p1(),  // 사용된 최대 위치 + 1
                cells.get_used(),     // 실제 사용 중인 셀 수
                head,                 // 현재 head 위치
                get_size(),          // 전체 캐시 크기
                n_swa);              // SWA 윈도우 크기

        // 상세 디버그: KV cache 시각화
        if ((debug == 2 && n_swa > 0) || debug > 2) {
            std::string ss;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (cells.is_empty(i)) {
                    ss += '.';  // 빈 셀
                } else {
                    assert(cells.seq_count(i) >= 1);

                    if (cells.seq_count(i) == 1) {
                        // 단일 시퀀스: 시퀀스 ID 표시
                        ss += std::to_string(cells.seq_get(i));
                    } else {
                        // 다중 시퀀스: 'M'으로 표시
                        ss += 'M';
                    }
                }
                // 줄바꿈 (256셀마다)
                if (i%256 == 255) {
                    ss += " *";
                    ss += '\n';
                }
            }
            LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
        }

        // 위치 정보 디버그
        if ((debug == 2 && n_swa > 0) || debug > 2) {
            std::string ss;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                std::string cur;
                if (cells.is_empty(i)) {
                    cur = '.';
                } else {
                    cur = std::to_string(cells.pos_get(i));
                }
                
                // 5자리로 정렬
                const int n = cur.size();
                for (int j = 0; j < 5 - n; ++j) {
                    cur += ' ';
                }
                ss += cur;
                
                if (i%256 == 255) {
                    ss += " *";
                }
                if (i%64 == 63) {
                    ss += '\n';
                }
            }
            LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
        }

        // 시퀀스별 위치 범위 출력
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (cells.seq_pos_min(s) < 0) {
                continue;
            }

            LLAMA_LOG_DEBUG("%s: min[%d] = %5d, max[%d] = %5d\n", __func__, 
                    s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
        }
    }

    // ==================== 메인 검색 루프 ====================
    
    uint32_t n_tested = 0;  // 무한 루프 방지용 카운터

    while (true) {
        // 캐시 끝을 넘어가면 처음부터 다시 검색 (순환)
        if (head_cur + n_tokens > cells.size()) {
            n_tested += cells.size() - head_cur;
            head_cur = 0;
            continue;
        }

        bool found = true;
        
        // ==================== 연속된 n_tokens 개 셀 검사 ====================
        
        for (uint32_t i = 0; i < n_tokens; i++) {
            //const llama_pos    pos    = ubatch.pos[i];      // 배치할 위치
            //const llama_seq_id seq_id = ubatch.seq_id[i][0]; // 시퀀스 ID

            // 이 셀을 사용할 수 있는가?
            bool can_use = cells.is_empty(head_cur + i);  // 1순위: 빈 셀

            if (!can_use && cells.seq_count(head_cur + i) == 1) {
                // 2순위: 단일 시퀀스 셀의 재사용 가능성 검사
                const llama_pos pos_cell = cells.pos_get(head_cur + i);

                // ==================== Causal Mask 검사 (비활성화됨) ====================
                // 주의: 현재는 비활성화됨. 미리 "미래" 토큰들을 정리하는 것이 좋음
                //if (cells.seq_has(head_cur + i, seq_id)) {
                //    can_use = pos_cell >= pos;  // 인과적 제약 만족하면 사용 가능
                //}

                if (!can_use) {
                    const llama_seq_id seq_id_cell = cells.seq_get(head_cur + i);

                    // ==================== SWA Mask 검사 ====================
                    // Sliding Window Attention에서 재사용 가능한지 확인
                    if (is_masked_swa(pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                        can_use = true;
                    }
                }
            }

            if (!can_use) {
                // 이 위치는 사용할 수 없음 - 다음 위치부터 재검색
                found = false;
                head_cur += i + 1;  // 실패한 위치 다음부터
                n_tested += i + 1;
                break;
            }
        }

        if (found) {
            // 성공: 연속된 n_tokens 개 셀을 모두 사용할 수 있음
            break;
        }

        // ==================== 무한 루프 방지 ====================

        if (n_tested >= cells.size()) {
            // 전체 캐시를 다 검사했지만 공간을 찾지 못함
            // SSD Offloading 핵심 지점:
            // 1. 메모리 부족 확실히 감지
            // 2. 오래된 KV cache를 SSD로 이동
            // 3. 필요한 KV cache를 SSD에서 로드
            // 4. 재시도 또는 배치 크기 축소
            //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
            return -1;
        }
    }

    return head_cur;  // 성공: 배치할 시작 위치 반환
}

/**
 * Micro-batch를 KV cache에 실제로 배치
 * 
 * 이 함수는 find_slot()으로 찾은 위치에 micro-batch를 실제로 배치합니다.
 * 메타데이터를 업데이트하고, 충돌하는 기존 데이터를 정리합니다.
 * 
 * 핵심 동작:
 * 1. 기존 데이터와의 충돌 검사 및 정리
 * 2. 새로운 토큰들의 위치와 시퀀스 ID 설정
 * 3. SWA 정책에 따른 오래된 데이터 정리
 * 4. Head 포인터 업데이트
 * 
 * @param head_cur: 배치할 시작 위치 (find_slot()의 결과)
 * @param ubatch: 배치할 micro-batch
 * 
 * SSD Offloading 연동 지점:
 * - 기존 데이터 정리 시 SSD로 백업
 * - 새 데이터 배치 완료 시 사용 패턴 추적 시작
 * - LRU/LFU 메타데이터 업데이트
 */
void llama_kv_cache_unified::apply_ubatch(uint32_t head_cur, const llama_ubatch & ubatch) {
    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: ubatch info:\n", __func__);
        LLAMA_LOG_DEBUG("%s:   n_tokens = %d, equal_seqs = %d\n", __func__, ubatch.n_tokens, ubatch.equal_seqs);
        LLAMA_LOG_DEBUG("%s:   n_seq_tokens = %d, n_seqs = %d\n", __func__, ubatch.n_seq_tokens, ubatch.n_seqs);
    }

    // ==================== SWA 정책을 위한 덮어쓸 데이터 추적 ====================
    
    // SWA (Sliding Window Attention)에서 덮어쓰게 될 최대 위치 추적
    // 나중에 일관성 유지를 위해 해당 범위의 오래된 데이터를 정리해야 함
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;  // -1 = 해당 시퀀스에서 제거할 데이터 없음
    }

    // ==================== 토큰별 배치 처리 ====================
    
    // ubatch는 시퀀스별로 구조화되어 있음:
    // - n_seqs: 시퀀스 개수
    // - n_seq_tokens: 시퀀스당 토큰 개수
    // - 총 토큰 수 = n_seqs * n_seq_tokens
    for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
        for (uint32_t j = 0; j < ubatch.n_seq_tokens; ++j) {
            const uint32_t idx = s*ubatch.n_seq_tokens + j;  // 전체 배열에서의 인덱스

            // ==================== 기존 데이터 충돌 처리 ====================

            if (!cells.is_empty(head_cur + idx)) {
                // 이 위치에 이미 데이터가 있음 (재사용 케이스)
                assert(cells.seq_count(head_cur + idx) == 1);  // 단일 시퀀스여야 함

                const llama_seq_id seq_id = cells.seq_get(head_cur + idx);
                const llama_pos    pos    = cells.pos_get(head_cur + idx);

                // SWA 정책: 덮어쓸 데이터의 최대 위치 기록
                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                // 기존 데이터 제거 (메타데이터만, 실제 텐서는 덮어쓰기됨)
                // SSD Offloading 지점: 중요한 데이터면 SSD에 백업
                cells.rm(head_cur + idx);
            }

            // ==================== 새 데이터 배치 ====================
            
            // 토큰 위치 설정
            cells.pos_set(head_cur + idx, ubatch.pos[idx]);

            // 시퀀스 ID 설정 (하나의 셀이 여러 시퀀스에 속할 수 있음)
            // TODO: fix indexing [UBATCH_IDX] - 향후 개선 필요
            for (int32_t i = 0; i < ubatch.n_seq_id[s]; i++) {
                cells.seq_add(head_cur + idx, ubatch.seq_id[s][i]);
            }
        }
    }

    // ==================== SWA 일관성 유지를 위한 정리 ====================
    
    // SWA 정책에서 중요한 불변 조건:
    // 각 시퀀스에서 [pos_min, pos_max] 범위의 모든 위치가 캐시에 존재해야 함
    // 
    // 문제: 새로 배치한 데이터가 기존 데이터를 덮어쓰면서 중간에 구멍이 생길 수 있음
    // 해결: 덮어쓴 위치보다 앞의 모든 데이터를 제거하여 일관성 유지
    //
    // 참고: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    
    for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;  // 이 시퀀스에서는 덮어쓴 데이터가 없음
        }

        // 시퀀스의 최소 위치가 덮어쓴 최대 위치보다 작거나 같으면 정리 필요
        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            // 해당 범위의 모든 데이터 제거
            // SSD Offloading 지점: 제거 전에 중요한 데이터는 SSD에 백업
            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }
    
    // ==================== Head 포인터 업데이트 ====================
    
    // 다음 배치를 위해 head를 배치된 영역의 끝으로 이동
    // 이렇게 하면 연속된 배치들이 메모리상에서도 연속적으로 배치됨 (지역성 향상)
    head = head_cur + ubatch.n_tokens;
    
    // SSD Offloading 최적화 지점:
    // - 새로 배치된 데이터의 접근 패턴 추적 시작
    // - 사용 빈도 메타데이터 초기화
    // - 백그라운드 정리 작업 스케줄링
}

bool llama_kv_cache_unified::get_can_shift() const {
    return true;
}

uint32_t llama_kv_cache_unified::get_size() const {
    return cells.size();
}

bool llama_kv_cache_unified::get_has_shift() const {
    return cells.get_has_shift();
}

uint32_t llama_kv_cache_unified::get_n_kv() const {
    return std::min(cells.size(), std::max(n_pad, GGML_PAD(cells.used_max_p1(), n_pad)));
}

ggml_tensor * llama_kv_cache_unified::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    return ggml_view_3d(ctx, k,
            hparams.n_embd_head_k, hparams.n_head_kv(il), n_kv,
            ggml_row_size(k->type, hparams.n_embd_head_k),
            ggml_row_size(k->type, hparams.n_embd_k_gqa(il)),
            0);
}

ggml_tensor * llama_kv_cache_unified::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_3d(ctx, v,
                hparams.n_embd_head_v, hparams.n_head_kv(il), n_kv,
                ggml_row_size(v->type, hparams.n_embd_head_v),    // v->nb[1]
                ggml_row_size(v->type, hparams.n_embd_v_gqa(il)), // v->nb[2]
                0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_3d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v,
            ggml_row_size(v->type, v->ne[1]*hparams.n_embd_head_v), // v->nb[1]
            ggml_row_size(v->type, v->ne[1]),                       // v->nb[2]
            0);
}

ggml_tensor * llama_kv_cache_unified::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il, uint32_t head_cur) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const int64_t n_tokens = k_cur->ne[2];

    ggml_tensor * k_view = ggml_view_1d(ctx, k,
            n_tokens*hparams.n_embd_k_gqa(il),
            ggml_row_size(k->type, hparams.n_embd_k_gqa(il))*head_cur);

    return ggml_cpy(ctx, k_cur, k_view);
}

ggml_tensor * llama_kv_cache_unified::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, int32_t il, uint32_t head_cur) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_tokens = v_cur->ne[2];

    v_cur = ggml_reshape_2d(ctx, v_cur, hparams.n_embd_v_gqa(il), n_tokens);

    ggml_tensor * v_view = nullptr;

    if (!v_trans) {
        v_view = ggml_view_1d(ctx, v,
                n_tokens*hparams.n_embd_v_gqa(il),
                ggml_row_size(v->type, hparams.n_embd_v_gqa(il))*head_cur);
    } else {
        // note: the V cache is transposed when not using flash attention
        v_view = ggml_view_2d(ctx, v, n_tokens, hparams.n_embd_v_gqa(il),
                (v->ne[1])*ggml_element_size(v),
                (head_cur)*ggml_element_size(v));

        v_cur = ggml_transpose(ctx, v_cur);
    }

    return ggml_cpy(ctx, v_cur, v_view);
}

void llama_kv_cache_unified::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens     = ubatch->n_tokens;
    const uint32_t n_seq_tokens = ubatch->n_seq_tokens;
    const uint32_t n_seqs       = ubatch->n_seqs;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    float * data = (float *) dst->data;

    const int64_t n_kv = dst->ne[0];

    // Use only the previous KV cells of the correct sequence for each token of the ubatch.
    // It's assumed that if a token in the batch has multiple sequences, they are equivalent.
    // Example with a cache of 10 tokens, 2 tokens populated in cache and 3 tokens in batch:
    //   Causal mask:
    //      xxx-------
    //      xxxx------
    //      xxxxx-----
    //   Non-causal mask:
    //      xxxxx-----
    //      xxxxx-----
    //      xxxxx-----
    // To visualize the mask, see https://github.com/ggml-org/llama.cpp/pull/12615
    for (uint32_t h = 0; h < 1; ++h) {
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch->seq_id[s][0];

            for (uint32_t j = 0; j < n_seq_tokens; ++j) {
                const uint32_t idx = s*n_seq_tokens + j;

                const llama_pos p1 = ubatch->pos[idx];

                for (uint32_t i = 0; i < n_kv; ++i) {
                    float f = 0.0f;

                    bool masked = false;

                    if (cells.is_empty(i)) {
                        masked = true;
                    } else {
                        const llama_pos p0 = cells.pos_get(i);

                        // mask the token if not the same sequence
                        masked = masked || (!cells.seq_has(i, seq_id));

                        // mask future tokens
                        masked = masked || (causal_attn && p0 > p1);

                        // apply SWA if any
                        masked = masked || (is_masked_swa(p0, p1));

                        if (!masked && hparams.use_alibi) {
                            f = -std::abs(p0 - p1);
                        }
                    }

                    if (masked) {
                        f = -INFINITY;
                    }

                    data[h*(n_kv*n_tokens) + idx*n_kv + i] = f;
                }
            }
        }

        // mask padded tokens
        if (data) {
            for (uint32_t j = n_tokens; j < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++j) {
                for (uint32_t i = 0; i < n_kv; ++i) {
                    data[h*(n_kv*n_tokens) + j*n_kv + i] = -INFINITY;
                }
            }
        }
    }
}

void llama_kv_cache_unified::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t i = 0; i < cells.size(); ++i) {
        data[i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
    }
}

void llama_kv_cache_unified::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_kv; ++i) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(i) ? -1 : cells.pos_get(i);

                data[h*(n_kv*n_tokens) + j*n_kv + i] = llama_relative_position_bucket(p0, ubatch->pos[j], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

size_t llama_kv_cache_unified::total_size() const {
    size_t size = 0;

    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache_unified::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache_unified::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += ggml_nbytes(layer.v);
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache_unified::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast  = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow  = cparams.yarn_beta_slow;

    const auto & n_rot     = hparams.n_rot;
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;

    // See llm_build_deepseek2() for why attn_factor has to be scaled for YaRN RoPE to work correctly.
    // See https://github.com/ggerganov/llama.cpp/discussions/7416 for detailed explanation.
    const float yarn_attn_factor = model.arch == LLM_ARCH_DEEPSEEK2
                                    ? 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale))
                                    : cparams.yarn_attn_factor;

    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache_unified * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size]

    const llama_kv_cache_unified * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }
}

llm_graph_result_ptr llama_kv_cache_unified::build_graph_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_cgraph * gf) const {
    auto res = std::make_unique<llm_graph_result>();

    const auto & n_embd_head_k = hparams.n_embd_head_k;
  //const auto & n_embd_head_v = hparams.n_embd_head_v;

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cells.size());
    ggml_set_input(inp->k_shift);

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_embd_head_k, n_head_kv, cells.size(),
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                0);

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, rope_factors, freq_base_l, freq_scale_l);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return res;
}

llm_graph_result_ptr llama_kv_cache_unified::build_graph_defrag(
                const llama_cparams & cparams,
                       ggml_context * ctx,
                        ggml_cgraph * gf,
                  const defrag_info & dinfo) const {
    auto res = std::make_unique<llm_graph_result>();

    const auto & ids = dinfo.ids;

#if 0
    // CPU defrag
    //
    // TODO: optimizations are possible:
    //       - multiple threads
    //       - avoid copying to the host memory when already there
    //
    // likely not worth the effort, as we have ggml_graph based defrag
    //

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa();

    const uint32_t kv_size = size;

    std::vector<uint8_t> buf_k;
    std::vector<uint8_t> buf_v;

    for (uint32_t il = 0; il < n_layer; ++il) {
        const size_t k_size_row = ggml_row_size(k_l[il]->type, n_embd_k_gqa);
        const size_t k_size     = ggml_row_size(k_l[il]->type, n_embd_k_gqa*kv_size);

        const size_t v_size_el = ggml_type_size(v_l[il]->type);
        const size_t v_size    = ggml_row_size (v_l[il]->type, n_embd_v_gqa*kv_size);

        buf_k.resize(k_size);
        buf_v.resize(v_size);

        ggml_backend_tensor_get(k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_get(v_l[il], buf_v.data(), 0, buf_v.size());

        // batch move [i, i+nm) to [id, id+nm)
        // note: cells can move only to a lower index
        for (uint32_t i = 0; i < n_kv; ++i) {
            const uint32_t id = ids[i];

            if (i == id || id == n_kv) {
                continue;
            }

            uint32_t nm = 1;

            while (i + nm < n_kv && ids[i + nm] == id + nm) {
                nm++;
            }

            // move keys
            {
                const int64_t os =  i*k_size_row;
                const int64_t od = id*k_size_row;

                memcpy(buf_k.data() + od, buf_k.data() + os, nm*k_size_row);
            }

            // move values (note: they are transposed)
            {
                const int64_t os =  i;
                const int64_t od = id;

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    memcpy(buf_v.data() + (od + j*kv_size)*v_size_el, buf_v.data() + (os + j*kv_size)*v_size_el, nm*v_size_el);
                }
            }

            i += nm - 1;
        }

        ggml_backend_tensor_set(k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_set(v_l[il], buf_v.data(), 0, buf_v.size());
    }
#else
    for (uint32_t i = 0; i < ids.size(); ++i) {
        const uint32_t id = ids[i];

        if (i == id || id == ids.size()) {
            continue;
        }

        uint32_t nm = 1;

        while (i + nm < ids.size() && ids[i + nm] == id + nm) {
            nm++;
        }

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
            const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            ggml_tensor * view_k_src = ggml_view_2d(ctx, layer.k,
                    n_embd_k_gqa, nm,
                    ggml_row_size(layer.k->type, n_embd_k_gqa),
                    ggml_row_size(layer.k->type, n_embd_k_gqa*i));

            ggml_tensor * view_k_dst = ggml_view_2d(ctx, layer.k,
                    n_embd_k_gqa, nm,
                    ggml_row_size(layer.k->type, n_embd_k_gqa),
                    ggml_row_size(layer.k->type, n_embd_k_gqa*id));

            ggml_tensor * view_v_src;
            ggml_tensor * view_v_dst;

            if (cparams.flash_attn) {
                // NOTE: the V cache is not transposed when using flash attention
                view_v_src = ggml_view_2d(ctx, layer.v,
                        n_embd_v_gqa, nm,
                        ggml_row_size(layer.v->type, n_embd_v_gqa),
                        ggml_row_size(layer.v->type, n_embd_v_gqa*i));

                view_v_dst = ggml_view_2d(ctx, layer.v,
                        n_embd_v_gqa, nm,
                        ggml_row_size(layer.v->type, n_embd_v_gqa),
                        ggml_row_size(layer.v->type, n_embd_v_gqa*id));
            } else {
                view_v_src = ggml_view_2d(ctx, layer.v,
                        nm, n_embd_v_gqa,
                        ggml_row_size(layer.v->type, cells.size()),
                        ggml_row_size(layer.v->type, i));

                view_v_dst = ggml_view_2d(ctx, layer.v,
                        nm, n_embd_v_gqa,
                        ggml_row_size(layer.v->type, cells.size()),
                        ggml_row_size(layer.v->type, id));
            }

            ggml_build_forward_expand(gf, ggml_cpy(ctx, view_k_src, view_k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, view_v_src, view_v_dst));
        }

        i += nm - 1;
    }

    //LLAMA_LOG_INFO("gf->n_nodes = %d\n", gf->n_nodes);
#endif

    return res;
}

/**
 * KV cache 조각화 해제 준비 - 이동 계획 수립
 * 
 * 메모리 조각화가 발생하면 성능이 저하되므로, 사용 중인 KV cache들을 
 * 메모리 앞쪽으로 압축하여 연속적으로 배치합니다.
 * 
 * 동작 원리:
 * 1. 메모리 앞쪽부터 스캔하여 빈 공간(hole) 찾기
 * 2. 메모리 뒤쪽부터 스캔하여 이동할 데이터 찾기
 * 3. 빈 공간에 맞는 크기의 연속된 데이터 블록 매칭
 * 4. 이동 계획(move plan) 생성
 * 5. 최대 이동 횟수 제한으로 성능 보장
 * 
 * @param n_max_nodes: 그래프의 최대 노드 수 (이동 횟수 제한용)
 * @return: 이동 계획 정보 (빈 객체면 이동 불필요)
 * 
 * SSD Offloading 연동:
 * - 조각화 해제 시 일시적으로 메모리 사용량 증가
 * - 이 시점에서 SSD로 임시 이동 고려
 * - 이동 완료 후 성능 향상으로 SSD 압박 감소
 */
llama_kv_cache_unified::defrag_info llama_kv_cache_unified::defrag_prepare(int32_t n_max_nodes) const {
    const uint32_t n_layer = layers.size();

    const uint32_t n_kv   = cells.used_max_p1();  // 사용된 최대 위치 + 1
    const uint32_t n_used = cells.get_used();     // 실제 사용 중인 셀 수

    assert(n_used <= n_kv);

    //const int64_t t_start = ggml_time_us();  // 성능 측정용 (주석 처리됨)

    // ==================== 이동 계획 제약 조건 계산 ====================
    
    uint32_t n_moves = 0;  // 실제 이동 횟수

    // 각 이동은 6*n_layer개의 텐서 연산이 필요:
    // - source view, destination view, copy operation
    // - Key, Value 각각에 대해
    // TODO: 임시 수정 https://github.com/ggerganov/llama.cpp/issues/6685#issuecomment-2057579516
    const uint32_t max_moves = (n_max_nodes - 2*n_layer)/(6*n_layer);

    // ==================== 이동 계획 데이터 구조 ====================
    
    defrag_info res;
    auto & ids = res.ids;

    // ids[i] = j: i번째 셀을 j번째 위치로 이동
    // ids[i] = n_kv: i번째 셀은 사용 중이지 않음 (이동 안함)
    ids.resize(n_kv, n_kv);

    // ==================== 메인 조각화 해제 알고리즘 ====================

    for (uint32_t i0 = 0; i0 < n_used; ++i0) {
        if (!cells.is_empty(i0)) {
            // 이미 사용 중인 셀 - 그대로 유지
            ids[i0] = i0;
            continue;
        }

        // ==================== 빈 공간(hole) 발견 - 채울 데이터 찾기 ====================

        uint32_t nh = 1;  // hole 크기

        // 연속된 빈 공간의 크기 계산
        while (i0 + nh < n_used && cells.is_empty(i0 + nh)) {
            nh++;
        }

        uint32_t nf = 0;  // 찾은 데이터 블록 크기
        uint32_t is = n_kv - 1;  // 뒤에서부터 검색 시작

        // ==================== 뒤쪽에서 연속된 데이터 블록 찾기 ====================
        
        // 뒤에서부터 nh개의 연속된 비어있지 않은 셀을 찾기
        for (; is > i0; --is) {
            if (cells.is_empty(is) || ids[is] != n_kv) {
                continue;  // 빈 셀이거나 이미 이동 계획이 있는 셀
            }

            // 사용 중이고 아직 이동 계획이 없는 셀
            nf++;

            if (nf == nh) {
                break;  // 필요한 만큼 찾음
            }
        }

        // 이것은 n_used가 정확하지 않을 때만 발생할 수 있음 (버그)
        GGML_ASSERT(nf == nh && "KV defrag bug: nf != nh");

        nf = 0;
        uint32_t i1 = is;

        // ==================== 이동 최적화: 연속된 블록 감지 ====================

        bool cont = false;  // 현재 연속된 블록을 이동 중인가?
        bool stop = false;  // 최대 이동 횟수 도달로 중단해야 하는가?

        // 찾은 데이터들을 빈 공간으로 이동 계획 수립
        for (; i1 < n_kv; ++i1) {
            if (cells.is_empty(i1) || ids[i1] != n_kv) {
                if (n_moves == max_moves) {
                    stop = true;
                    break;
                }

                cont = false;
                continue;
            }

            // 이 셀을 (i0 + nf) 위치로 이동
            ids[i1] = i0 + nf;

            if (!cont) {
                // 새로운 연속 블록 시작
                n_moves++;
                cont = true;
            }

            nf++;

            if (nf == nh) {
                break;  // 빈 공간을 모두 채움
            }
        }

        if (stop || n_moves == max_moves) {
            break;  // 최대 이동 횟수 도달
        }

        //LLAMA_LOG_INFO("(tmp log) KV defrag: move [%u, %u) to [%u, %u)\n", is, i1 + 1, i0, i0 + nh);

        i0 += nh - 1;  // 다음 iteration에서 ++i0 되므로 -1
    }

    if (n_moves == 0) {
        // 이동할 필요 없음
        return {};
    }

    LLAMA_LOG_DEBUG("%s: (tmp log) KV defrag cell moves: %u\n", __func__, n_moves);
    LLAMA_LOG_DEBUG("%s: expected gf nodes: %u\n", __func__, 6*n_moves*n_layer);

    return res;
}

bool llama_kv_cache_unified::is_masked_swa(llama_pos p0, llama_pos p1) const {
    assert(p0 >= 0 && p1 >= 0);

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:
            {
            } break;
        case LLAMA_SWA_TYPE_STANDARD:
            {
                if (p1 - p0 >= (int32_t) n_swa) {
                    return true;
                }
            } break;
        case LLAMA_SWA_TYPE_CHUNKED:
            {
                const llama_pos pos_chunk_start = (p1 / n_swa) * n_swa;

                if (p0 < pos_chunk_start) {
                    return true;
                }
            } break;
    }

    return false;
}

/**
 * KV cache 상태를 파일에 저장
 * 
 * 사용자 세션의 KV cache를 디스크에 저장하여 나중에 복원할 수 있게 합니다.
 * 특정 시퀀스만 저장하거나 전체 캐시를 저장할 수 있습니다.
 * 
 * 저장 과정:
 * 1. 저장할 셀들의 범위 계산
 * 2. 메타데이터 저장 (위치, 시퀀스 ID 정보)
 * 3. 실제 텐서 데이터 저장 (Key, Value)
 * 
 * @param io: 출력 스트림 인터페이스
 * @param seq_id: 저장할 시퀀스 ID (-1이면 전체 캐시)
 * 
 * SSD Offloading 연관성:
 * - 이미 SSD에 있는 데이터는 직접 복사 가능
 * - 메모리에서 SSD로의 영구 이동과 유사한 동작
 * - 압축 기법 적용 가능
 */
void llama_kv_cache_unified::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    // ==================== 저장 대상 셀 범위 계산 ====================
    
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // 연속된 셀 범위들 (시작, 끝)
    uint32_t cell_count = 0;  // 총 셀 개수

    // 지정된 시퀀스(또는 전체)에 해당하는 모든 셀을 찾고 연속된 범위로 그룹화
    uint32_t cell_range_begin = cells.size();  // 현재 범위의 시작점 (초기값: 유효하지 않은 값)

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && (seq_id == -1 || cells.seq_has(i, seq_id))) {
            // 저장 대상 셀 발견
            ++cell_count;
            
            if (cell_range_begin == cells.size()) {
                // 새로운 범위 시작
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != cells.size()) {
                // 현재 범위 종료
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = cells.size();  // 범위 리셋
            }
        }
    }

    // 마지막 범위 처리
    if (cell_range_begin != cells.size()) {
        cell_ranges.emplace_back(cell_range_begin, cells.size());
    }

    // ==================== 디버그 검증 ====================
    
    // 범위별 셀 개수의 합이 전체 셀 개수와 일치하는지 확인
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    // ==================== 실제 저장 수행 ====================
    
    // 1. 셀 개수 저장 (헤더)
    io.write(&cell_count, sizeof(cell_count));

    // 2. 메타데이터 저장
    state_write_meta(io, cell_ranges, seq_id);
    
    // 3. 실제 텐서 데이터 저장
    state_write_data(io, cell_ranges);
}

/**
 * 파일에서 KV cache 상태를 복원
 * 
 * 이전에 저장한 KV cache를 메모리로 로드하여 세션을 복원합니다.
 * 
 * 복원 과정:
 * 1. 셀 개수 읽기
 * 2. 메타데이터 복원 및 적절한 위치 할당
 * 3. 실제 텐서 데이터 로드
 * 
 * @param io: 입력 스트림 인터페이스
 * @param seq_id: 복원할 시퀀스 ID (-1이면 전체 캐시)
 * 
 * 실패 시 동작:
 * - 지정된 시퀀스만 삭제 (seq_id >= 0)
 * - 전체 캐시 초기화 (seq_id == -1)
 * - 예외 발생
 */
void llama_kv_cache_unified::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    uint32_t cell_count;
    io.read_to(&cell_count, sizeof(cell_count));

    bool res = true;
    
    // 순차적으로 복원 수행
    res = res && state_read_meta(io, cell_count, seq_id);  // 메타데이터 복원
    res = res && state_read_data(io, cell_count);          // 텐서 데이터 복원

    if (!res) {
        // ==================== 복원 실패 시 정리 ====================
        
        if (seq_id == -1) {
            // 전체 복원 실패 - 캐시 완전 초기화
            clear(true);
        } else {
            // 특정 시퀀스 복원 실패 - 해당 시퀀스만 제거
            seq_rm(seq_id, -1, -1);
        }
        throw std::runtime_error("failed to restore kv cache");
    }
}

/**
 * KV cache 메타데이터를 파일에 저장
 * 
 * 각 셀의 위치 정보와 시퀀스 ID 정보를 저장합니다.
 * 실제 텐서 데이터가 아닌 관리 정보만 저장합니다.
 * 
 * @param io: 출력 스트림
 * @param cell_ranges: 저장할 셀 범위들
 * @param seq_id: 필터링할 시퀀스 ID (-1이면 모든 시퀀스)
 */
void llama_kv_cache_unified::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            // ==================== 시퀀스 ID 필터링 ====================
            
            std::vector<llama_seq_id> seq_ids;

            // 이 셀이 속한 시퀀스들 중 저장 대상인 것들만 수집
            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            // ==================== 셀 정보 저장 ====================

            const llama_pos pos     = cells.pos_get(i);    // 이 셀의 토큰 위치
            const uint32_t n_seq_id = seq_ids.size();      // 속한 시퀀스 개수

            // 위치 정보 저장
            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            // 시퀀스 ID들 저장
            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache_unified::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {
    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    std::vector<uint8_t> tmp_buf;

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        // Write key type
        const int32_t k_type_i = (int32_t)layer.k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(layer.k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length each into tmp_buf and write out
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(layer.k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            // Write value type
            const int32_t v_type_i = (int32_t)layer.v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(layer.v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length each into tmp_buf and write out
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(layer.v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            // Write value type
            const int32_t v_type_i = (int32_t)layer.v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(layer.v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length each into tmp_buf and write out
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(layer.v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache_unified::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence

        seq_rm(dest_seq_id, -1, -1);

        llama_sbatch sbatch;
        llama_ubatch ubatch = sbatch.reserve_ubatch(cell_count, /* has_embd */ false);

        ubatch.n_tokens = cell_count;
        ubatch.n_seq_tokens = cell_count;
        ubatch.n_seqs = 1;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        const auto head_cur = find_slot(ubatch);
        if (head_cur < 0) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        apply_ubatch(head_cur, ubatch);

        // keep the head at the old position because we will read the KV data into it in state_read_data()
        head = head_cur;

        // DEBUG CHECK: head_cur should be our first cell, head_cur + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head_cur + cell_count <= cells.size());
        GGML_ASSERT(cells.pos_get(head_cur)                  == ubatch.pos[0]);
        GGML_ASSERT(cells.pos_get(head_cur + cell_count - 1) == ubatch.pos[cell_count - 1]);
        GGML_ASSERT(cells.seq_has(head_cur,                  dest_seq_id));
        GGML_ASSERT(cells.seq_has(head_cur + cell_count - 1, dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache_unified::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t v_trans;
    uint32_t n_layer;

    io.read_to(&v_trans, sizeof(v_trans));
    io.read_to(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        // Read type of key
        int32_t k_type_i_ref;
        io.read_to(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) layer.k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read_to(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(layer.k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            ggml_backend_tensor_set(layer.k, io.read(cell_count * k_size_row), head * k_size_row, cell_count * k_size_row);
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t)layer.v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read_to(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(layer.v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                ggml_backend_tensor_set(layer.v, io.read(cell_count * v_size_row), head * v_size_row, cell_count * v_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t)layer.v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read_to(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(layer.v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read_to(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    const size_t dst_offset = (head + j * cells.size()) * v_size_el;
                    ggml_backend_tensor_set(layer.v, io.read(cell_count * v_size_el), dst_offset, cell_count * v_size_el);
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_unified_state
//

llama_kv_cache_unified_state::llama_kv_cache_unified_state(llama_memory_status status) : status(status) {}

llama_kv_cache_unified_state::llama_kv_cache_unified_state(
        llama_kv_cache_unified * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();
    head = 0;
}

llama_kv_cache_unified_state::llama_kv_cache_unified_state(
        llama_kv_cache_unified * kv,
        llama_context * lctx,
        bool do_shift,
        defrag_info dinfo) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), dinfo(std::move(dinfo)) {
    if (!do_shift && this->dinfo.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_unified_state::llama_kv_cache_unified_state(
        llama_kv_cache_unified * kv,
        llama_sbatch sbatch,
        llama_kv_cache_unified::ubatch_heads heads,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sbatch(std::move(sbatch)), heads(std::move(heads)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_unified_state::~llama_kv_cache_unified_state() = default;

bool llama_kv_cache_unified_state::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_unified_state::apply() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, dinfo);

        return true;
    }

    kv->apply_ubatch(heads[i_next], ubatches[i_next]);

    n_kv = kv->get_n_kv();
    head = heads[i_next];

    return true;
}

std::vector<int64_t> & llama_kv_cache_unified_state::out_ids() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return sbatch.out_ids;
}

llama_memory_status llama_kv_cache_unified_state::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_unified_state::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint32_t llama_kv_cache_unified_state::get_n_kv() const {
    return n_kv;
}

ggml_tensor * llama_kv_cache_unified_state::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv);
}

ggml_tensor * llama_kv_cache_unified_state::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv);
}

ggml_tensor * llama_kv_cache_unified_state::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, il, head);
}

ggml_tensor * llama_kv_cache_unified_state::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, il, head);
}

void llama_kv_cache_unified_state::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_unified_state::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_unified_state::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

uint32_t llama_kv_cache_unified::get_padding(const llama_cparams & cparams) {
    // the FA kernels require padding to avoid extra runtime boundary checks
    return cparams.flash_attn ? 256u : 32u;
}

int32_t llama_kv_cache_unified_state::get_head() const {
    return head;
}
