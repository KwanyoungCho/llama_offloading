#include "llama.h"
#include "../../src/llama-kv-offloading.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>


int main() {
    // 1) 캐시 경로 지정
    const char* cache_dir = "./kv_cache_test";
    // offloader 생성
    llama_kv_offloader* offloader = llama_kv_offloader_init(cache_dir);
    if (!offloader) {
        fprintf(stderr, "Failed to init offloader\n");
        return 1;
    }

    // 2) 테스트할 token 길이 목록
    // std::vector<int> token_lengths = { 32, 64, 96, 128, 160, 192, 224, 256, 288, 320,
    //                                      352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 
    //                                      672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 
    //                                      992, 1024,
    //                                      1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664,
    //                                      1728, 1792, 1856, 1920, 1984, 2048
    //                                     };

    std::vector<int> token_lengths = { 35, 35, 35, 35, 35 };

    // 하나 토큰당 K/V 크기: 4096 요소 * 2 바이트(half) = 8192 바이트
    const size_t per_token_bytes = 4096 * 2;

    double avg_load_time = 0;
    double avg_save_time = 0;
    
    for (int tokens : token_lengths) {
        for (int layer = 0; layer < 32; layer++) {
            size_t bytes = tokens * per_token_bytes;

            // 3) 임의 데이터 버퍼 생성
            std::vector<char> k_data(bytes, 0xAA);
            std::vector<char> v_data(bytes, 0x55);

            // 4) 비동기 저장 호출
            const auto t_save_start = ggml_time_us();
            // llama_kv_offloader_save_layer(
            //     offloader,
            //     layer,
            //     k_data.data(), v_data.data(),
            //     bytes,        bytes
            // );
            // 내부 printf 타이밍 로그가 찍힐 때까지 대기
            llama_kv_offloader_wait_all(offloader);
            const auto t_save_end = ggml_time_us();
            avg_save_time += (t_save_end - t_save_start) / 1000.0;

            // 5) 비동기 로드 테스트를 위해, 읽어들일 버퍼 준비
            std::vector<char> k_load(bytes);
            std::vector<char> v_load(bytes);
            
            const auto t_load_start = ggml_time_us();
            // llama_kv_offloader_load_layer(
            //     offloader,
            //     layer,
            //     k_load.data(), v_load.data(),
            //     bytes,         bytes
            // );
            llama_kv_offloader_wait_all(offloader);
            const auto t_load_end = ggml_time_us();
            
            avg_load_time += (t_load_end - t_load_start) / 1000.0;
        }
        avg_load_time /= 32;
        // printf("%d, %f\n", tokens, avg_load_time);
        avg_load_time = 0;
    }

    // 6) 정리
    llama_kv_offloader_free(offloader);
    return 0;
}