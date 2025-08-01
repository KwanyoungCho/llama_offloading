//Tree-based EAGLE 구현 코드
//Static Tree-based EAGLE을 우선적으로 구현한 후에 Dynamic Tree Generation 알고리즘을 추가할 계획입니다.
//-ym-

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include "../src/llama-model.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct callback_data { //callback function의 return 값을 저장할 구조체 선언 -ym-
    std::vector<float> data; //float 타입으로 변경 -ym-
};

static bool cb_get_hidden(struct ggml_tensor * tensor, bool ask, void * user_data) { //callback function -ym-
    if (ask) {
        static const char * result_norm_name = "result_norm";
        const bool is_result_norm = strcmp(tensor->name, result_norm_name) == 0;
        return is_result_norm;
    }

    //LOG_DBG("[%d, %d, %d, %d]\n", tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    auto * cb_data = (struct callback_data *) user_data;
    auto n_bytes = ggml_nbytes(tensor);
    cb_data->data.resize(n_bytes / sizeof(float)); //float 타입으로 변경 -ym-
    ggml_backend_tensor_get(tensor, cb_data->data.data(), 0, n_bytes);

    return true;
}

int64_t start_time;

static bool cb_get_latency(struct ggml_tensor * tensor, bool ask, [[maybe_unused]] void * user_data) { //latency profiling callback function -ym-
    if (ask) {
        start_time = ggml_time_us();
        return true;
    }

    int64_t end_time = ggml_time_us();
    int64_t latency = end_time - start_time;
    LOG_DBG("[[Latency for tensor]] '%s' (%s): %ld us ==> (%d)\n", tensor->name, ggml_op_name(tensor->op), latency, (int)ggml_backend_buffer_is_host(tensor->buffer));
    ggml_tensor * src_tensor = tensor->src[0];
    LOG_DBG("[[Latency for tensor]] [%d, %d, %d, %d]\n", (int)src_tensor->ne[0], (int)src_tensor->ne[1], (int)src_tensor->ne[2], (int)src_tensor->ne[3]);
    LOG_DBG("[[Latency for tensor]] [%d, %d, %d, %d]\n", (int)tensor->ne[0], (int)tensor->ne[1], (int)tensor->ne[2], (int)tensor->ne[3]);


    return true;
}

struct seq_draft { //각 드래프트 시퀀스(트리의 브랜치)의 상태를 저장하는 구조체 -ym-
    bool active   = false; //verification 단계에서 시퀀스가 활성화되었는지 여부 -ym-
    bool drafting = false; //drafting 단계에서 시퀀스가 활성화되었는지 여부 -ym-
    bool skip     = false; //drafting 단계에서 이 시퀀스를 건너뛸지 여부 -ym-

    int i_batch_dft = 0; //드래프트 모델의 배치에서 이 시퀀스의 마지막 토큰 인덱스 -ym-
    std::vector<int> i_batch_tgt; //타겟 모델의 배치에서 이 시퀀스에 해당하는 토큰들의 인덱스 -ym-

    std::vector<llama_token> tokens; //이 시퀀스가 추측한 토큰들의 목록 -ym-
    std::vector<std::vector<llama_token_data>> dists;

    struct common_sampler * smpl = nullptr;
};

int main(int argc, char ** argv) {
    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    if (params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
    // const float p_draft_split = params.speculative.p_split; // 사용하지 않음 -ym-

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    callback_data cb_data; //callback data 구조체 변수 선언 -ym-
    params.cb_eval = cb_get_hidden; //callback function 등록 -ym-
    params.cb_eval_user_data = &cb_data; //callback function의 return 값을 callback data 구조체 변수로 받음 -ym-

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    common_init_result llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();

    // load the draft model
    params.devices = params.speculative.devices;
    params.model = params.speculative.model;
    params.n_gpu_layers = params.speculative.n_gpu_layers;
    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    //params.cb_eval = cb_get_latency;
    common_init_result llama_init_dft = common_init_from_params(params);

    model_dft = llama_init_dft.model.get();
    ctx_dft   = llama_init_dft.context.get();

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);

    // Trick: if the output buffer is in host memory, we need to allocate a new buffer for the draft model
    if (ggml_backend_buffer_is_host(llama_get_model(ctx_dft)->output->buffer)) {
        void * data = malloc(ggml_nbytes(llama_get_model(ctx_tgt)->output));
        llama_get_model(ctx_dft)->output->data = data;
    }
    // copy output parameters from target to draft
    ggml_backend_tensor_copy(llama_get_model(ctx_tgt)->output, llama_get_model(ctx_dft)->output);

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);
    // target model sampling context (reuse the llama_context's sampling instance)
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    llama_batch temp_batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    int temp_n_past = 0;
    for (size_t i = 0; i < inp.size() - 1; i++) {
        common_batch_add(temp_batch_tgt, inp[i], temp_n_past++, { 0 }, true);
    }

    // eval the prompt with both models
    llama_decode(ctx_tgt, temp_batch_tgt);
    std::vector<float> sliced_data = std::vector<float>(cb_data.data.begin(), cb_data.data.end()); // callback data에서 마지막 데이터를 제외한 나머지 백업 -ym-

    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(), 1));
    std::vector<float> backup_data = std::vector<float>(cb_data.data.begin(), cb_data.data.end()); // callback data에서 마지막 데이터만 백업 -ym-

    llama_decode_eagle(ctx_dft, llama_batch_get_one(inp.data() + 1, n_input - 1), sliced_data.data());

    // float* p_data = sliced_data.data();
    // size_t total_size = sliced_data.size();
    // LOG("total_size: %d\n", total_size);
    // if (total_size == 0) {
    //     LOG("데이터가 비어있습니다.\n");
    // }
    // else {
    //     LOG("sliced 데이터 크기:  %d개\n", total_size / 4096);
    //     for (int i = 0; i < 10; ++i) {
    //         // cb_data.data[i]를 사용해 i번째 요소에 접근
    //         // uint8_t는 문자로 출력될 수 있으므로 int로 변환하여 숫자 값을 확인
    //         LOG("%lf ", *(p_data + i));
    //     }
    //     LOG("\n");
    //     size_t start_index = total_size - 10;
    //     for (int i = start_index; i < total_size; ++i) {
    //         LOG("%lf ", *(p_data + i));
    //     }
    //     LOG("\n");
    // }
    LOG("\n");LOG("\n");

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_vocab_n_tokens(model_dft));

    // how many tokens to draft each time
    int n_draft = params.speculative.n_max;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size() - 1;

    // used to determine end of generation
    bool has_eos = false;

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    // [추가] 각 단계별 수락 길이를 저장하기 위한 벡터
    std::vector<int> acceptance_lengths;

    for (int s = 0; s < n_seq_dft; ++s) {
        // allocate llama_sampler for each draft sequence
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    while (true) {
        std::set<int> active_seqs = {};

        // print current draft sequences
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) { //active 변수의 초기 값은 false, 따라서 첫 prefill 후에는 이 반복문 동작 안함 -ym-
                continue;
            }

            active_seqs.insert(s);
            const auto & tokens = drafts[s].tokens;

            LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str());
        }

        int i_dft  = 0;
        int s_keep = 0;

        llama_token token_id;
        std::string token_str;

        std::vector<float> temp2;
        std::vector<llama_token> recompute;

        // loop until we fail to accept a drafted token or we run out of drafted tokens
        while (true) {

            // check if the target token matches any of the drafts
            // for stochastic sampling, attempt to match the token with the drafted tokens
            {
                bool accept = false;
                if (params.sampling.temp > 0) {
                    // stochastic verification
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);

                    auto & dist_tgt = *common_sampler_get_candidates(smpl);

                    float p_tgt = 0.0f;
                    float p_dft = 0.0f;

                    while (active_seqs.size() > 0) {
                        // randomly select a sequence to verify from active sequences
                        std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                        int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                        if (i_dft >= (int) drafts[s].tokens.size()) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                            continue;
                        }
                        if (accept) {
                            // if we already accepted a token, we can skip the rest
                            if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                            }
                            continue;
                        }

                        LOG_DBG("verifying sequence #%d at pos #%d from %d active sequence(s)\n", s, i_dft, (int) active_seqs.size());
                        float r = u_dist(rng);
                        llama_token_data_array dist_dft = { drafts[s].dists[i_dft].data() , drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true };

                        //GGML_ASSERT(dist_tgt.size <= dist_dft.size);

                        // acquire the token probabilities assigned by the draft and target models
                        for (size_t i = 0; i < dist_tgt.size; i++) {
                            if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                                p_tgt = dist_tgt.data[i].p;
                                break;
                            }
                        }
                        for (size_t i = 0; i < dist_dft.size; i++) {
                            if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                                p_dft = dist_dft.data[i].p;
                                break;
                            }
                        }
                        LOG_DBG("r = %f, p_dft = %f, p_tgt = %f\n", r, p_dft, p_tgt);
                        if (r <= p_tgt / p_dft) {
                            s_keep = s;
                            accept = true;
                            token_id = drafts[s].tokens[i_dft];
                            token_str = common_token_to_piece(ctx_tgt, token_id);
                            common_sampler_accept(smpl, token_id, true);

                            LOG_DBG("draft token %d of sequence %d (%d, '%s') accepted\n", i_dft, s, token_id, token_str.c_str());
                            break;
                        } else {
                            LOG_DBG("draft token %d of sequence %d (%d, '%s') rejected\n", i_dft, s, drafts[s].tokens[i_dft], common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str());
                            drafts[s].active = false;

                            // calculate residual probability
                            GGML_ASSERT(dist_tgt.sorted);
                            GGML_ASSERT(dist_dft.sorted);

                            // sort dist by id
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });
                            std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });

                            float sum_probs = 0.0f;

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                if (i < dist_dft.size) {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                                } else {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                                }

                                sum_probs += dist_tgt.data[i].p;
                            }

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p /= sum_probs;
                            }

                            // sort dist_tgt by p desc
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.p > b.p;
                            });
                        }

                        active_seqs.erase(s);
                        for (int i = 0; i < n_seq_dft; i++) {
                            if (i == s) {
                                continue;
                            }
                            if (drafts[i].active && drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                // synchronize active status for sequences with the same drafted token
                                drafts[i].active = drafts[i].active && accept;
                                if (!drafts[i].active) {
                                    active_seqs.erase(s);
                                }
                            }
                        }
                    }

                    if (!accept) {
                        // all drafted tokens were rejected
                        // sample from the target model
                        LOG_DBG("all drafted tokens were rejected, sampling from residual distribution\n");
                        std::vector<float> probs(dist_tgt.size);
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            probs[i] = dist_tgt.data[i].p;
                        }

                        std::discrete_distribution<> dist(probs.begin(), probs.end());

                        const int idx = dist(rng);

                        token_id = dist_tgt.data[idx].id;
                        common_sampler_accept(smpl, token_id, true);
                        token_str = common_token_to_piece(ctx_tgt, token_id);
                    }
                } else {
                    // greedy verification

                    // sample from the target model
                    LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);

                    common_sampler_accept(smpl, token_id, true);

                    token_str = common_token_to_piece(ctx_tgt, token_id);

                    temp2.insert(temp2.end(), backup_data.begin() + (4096 * (drafts[s_keep].i_batch_tgt[i_dft])), backup_data.begin() + (4096 * (drafts[s_keep].i_batch_tgt[i_dft] + 1)));
                    recompute.push_back(token_id);

                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) {
                            continue;
                        }

                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());

                            s_keep = s;
                            accept = true;
                        } else {
                            drafts[s].active = false;
                        }
                    }
                }

                if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                    has_eos = true;
                }
                ++n_predict;

                if (accept) {
                    ++n_accept;
                    ++n_past_tgt;
                    ++n_past_dft;
                    ++i_dft;
                    if (params.use_color) {
                        // Color token according to its origin sequence
                        LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                    } else {
                        LOG("%s", token_str.c_str());
                    }
                    continue;
                } else {
                    LOG("%s", token_str.c_str());
                    break;
                }
            }
        }

        // [추가] 현재 단계의 수락 길이를 저장합니다.
        // 루프가 끝났을 때 i_dft는 이번 단계에서 연속적으로 수락된 토큰의 개수와 같습니다.
        acceptance_lengths.push_back(i_dft);

        backup_data = temp2;
        std::vector temp3 = std::vector<float>(backup_data.end() - 4096, backup_data.end());
        int recompute_point = n_past_dft - i_dft;

        LOG_DBG("Current n_accept: %d, n_drafted: %d, n_predict: %d\n", n_accept, n_drafted, n_predict);

        {
            LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());

            // TODO: simplify
            {
                LOG_DBG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);

                llama_memory_seq_keep(mem_dft, s_keep);
                llama_memory_seq_cp  (mem_dft, s_keep, 0, -1, -1);
                llama_memory_seq_keep(mem_dft, 0);

                llama_memory_seq_rm  (mem_tgt, s_keep, n_past_tgt, -1);
                llama_memory_seq_keep(mem_tgt, s_keep);
                llama_memory_seq_cp  (mem_tgt, s_keep, 0, -1, -1);
                llama_memory_seq_keep(mem_tgt, 0);
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
            // note: will be erased after the speculation phase
            drafts[0].tokens.push_back(token_id);
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].i_batch_tgt.push_back(0);

            llama_memory_seq_rm(mem_dft, 0, recompute_point, -1);

            //recompute logic 추가 -ym-
            if (i_dft > 0) {
                std::vector temp4 = std::vector<float>(backup_data.begin(), backup_data.end() - 4096);

                common_batch_clear(batch_dft);
                for (size_t i = 0; i < recompute.size() - 1; i++) {
                    common_batch_add  (batch_dft, recompute[i], recompute_point + i, { 0 }, false);
                }
                llama_decode_eagle(ctx_dft, batch_dft, temp4.data());
            }

            common_batch_clear(batch_dft);
            common_batch_add(batch_dft, token_id, n_past_dft, {0}, true);

            LOG_DBG("n_past_tgt: %d, n_past_dft: %d\n", n_past_tgt, n_past_dft);
            LOG_DBG("recompute point: %d, n_past_dft: %d, recompute.size(): %zu, batch_dft.n_tokens: %d, backup_data.size(): %zu\n", recompute_point, n_past_dft, recompute.size(), batch_dft.n_tokens, backup_data.size()/4096);

            // LOG_DBG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());
            llama_decode_eagle(ctx_dft, batch_dft, temp3.data());
            ++n_past_dft;
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl);

        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        // sample n_draft tokens from the draft model using tree-based sampling
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            std::vector<float> temp; // callback data를 임시로 저장 -ym-

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }

                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);

                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl);

                for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            k, s, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                std::vector<int> sa(1, s);

                temp.insert(temp.end(), cb_data.data.begin() + (4096 * s), cb_data.data.begin() + (4096 * (s + 1)));

                // attempt to split the branch if the probability is high enough
                for (int f = 1; f < 2; ++f) {
                    LOG_DBG("cur_p->data[f].p = %lf\n", cur_p->data[f].p);
                    // if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) {
                    if (n_seq_cur < n_seq_dft) {
                        LOG_DBG("splitting seq %3d into %3d\n", s, n_seq_cur);

                        llama_memory_seq_rm(mem_dft,    n_seq_cur, -1, -1);
                        llama_memory_seq_cp(mem_dft, s, n_seq_cur, -1, -1);

                        LOG_DBG("디버그: n_seq_cur = %d, cb_data.data.size() = %zu\n", n_seq_cur, backup_data.size());
                        temp.insert(temp.end(), cb_data.data.begin() + (4096 * s), cb_data.data.begin() + (4096 * (s + 1)));

                        // all previous tokens from this branch are now also part of the new branch
                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }

                        // copy the draft state
                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true;

                        drafts[n_seq_cur].tokens      = drafts[s].tokens;
                        drafts[n_seq_cur].dists       = drafts[s].dists;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                        if (drafts[n_seq_cur].smpl) {
                            common_sampler_free(drafts[n_seq_cur].smpl);
                        }
                        drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);

                        sa.push_back(n_seq_cur);

                        n_seq_cur++;
                    } else {
                        break;
                    }
                }

                // add drafted token for each sequence
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p->data[is].id;

                    const int s = sa[is];

                    common_sampler_accept(drafts[s].smpl, id, true);

                    drafts[s].tokens.push_back(id);
                    // save cur_p.data into drafts[s].dists
                    drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});

                    // add unique drafted tokens to the target batch
                    drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);

                    common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);

                    // add the token to the batch for batched decoding with the draft model
                    drafts[s].i_batch_dft = batch_dft.n_tokens;

                    common_batch_add(batch_dft, id, n_past_cur, { s }, true);

                    if (batch_tgt.n_tokens > n_draft) {
                        drafts[s].drafting = false;
                    }
                }
            }

            // no sequence is drafting anymore
            if (batch_dft.n_tokens == 0) {
                break;
            }

            LOG_DBG("temp.size(): %zu, batch_dft.n_tokens: %d\n", temp.size()/(size_t)4096, batch_dft.n_tokens);

            // evaluate the drafted tokens on the draft model
            llama_decode_eagle(ctx_dft, batch_dft, temp.data());
            ++n_past_cur;
            ++n_drafted;

            if (batch_tgt.n_tokens > n_draft) {
                break;
            }
        }

        // evaluate the target model on the drafted tokens
        {
            llama_memory_seq_keep(mem_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
            }

            // LOG_DBG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            llama_decode(ctx_tgt, batch_tgt);
            backup_data = cb_data.data;
            ++n_past_tgt;
        }

        // the first token is always proposed by the target model before the speculation loop so we erase it here
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    // [추가] 수락 길이 통계 계산 및 출력
    if (!acceptance_lengths.empty()) {
        const double avg_len = std::accumulate(acceptance_lengths.begin(), acceptance_lengths.end(), 0.0) / acceptance_lengths.size();
        const int min_len = *std::min_element(acceptance_lengths.begin(), acceptance_lengths.end());
        const int max_len = *std::max_element(acceptance_lengths.begin(), acceptance_lengths.end());

        LOG_INF("\n");
        LOG_INF("Acceptance length stats:\n");
        LOG_INF("  Min length: %d\n", min_len);
        LOG_INF("  Max length: %d\n", max_len);
        LOG_INF("  Avg length: %.3f\n", avg_len);
    }

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    // TODO: print sampling/grammar timings for all drafts
    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_batch_free(batch_dft);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
