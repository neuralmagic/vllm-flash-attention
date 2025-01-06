#include "registration.h"
#include <torch/library.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

/**
 *  Externs for the flash_attn ops to be exposed as a pytorch library
 */

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               c10::optional<const at::Tensor> &leftpad_k_, // batch_size
               c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               c10::optional<at::Generator> gen_);

std::vector<at::Tensor>
mha_varlen_fwd_meta(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                    const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                    const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                    c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                    const at::Tensor &cu_seqlens_q,  // b+1
                    const at::Tensor &cu_seqlens_k,  // b+1
                    c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
                    c10::optional<const at::Tensor> &leftpad_k_, // batch_size
                    c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                    c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
                    int max_seqlen_q,
                    const int max_seqlen_k,
                    const float p_dropout,
                    const float softmax_scale,
                    const bool zero_tensors,
                    bool is_causal,
                    int window_size_left,
                    int window_size_right,
                    const float softcap,
                    const bool return_softmax,
                    c10::optional<at::Generator> gen_) {
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = q.sizes()[0];
    const int num_heads = q.sizes()[1];
    const int head_size = q.sizes()[2];

    auto opts = q.options();
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

    return {torch::empty_like(q), softmax_lse};
}

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                c10::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &seqlens_k_, // batch_size
                c10::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                c10::optional<const at::Tensor> &leftpad_k_, // batch_size
                c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits);


std::vector<at::Tensor>
mha_fwd_kvcache_meta(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                     const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                     const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                     c10::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                     c10::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                     c10::optional<const at::Tensor> &seqlens_k_, // batch_size
                     c10::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                     c10::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                     c10::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                     c10::optional<const at::Tensor> &leftpad_k_, // batch_size
                     c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                     c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                     c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                     const float softmax_scale,
                     bool is_causal,
                     int window_size_left,
                     int window_size_right,
                     const float softcap,
                     bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                     int num_splits) {
    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];

    auto opts = q.options();
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    auto out = torch::empty_like(q);

    return {out, softmax_lse};
}


/**
 * Unforunately, the type signatures of the flash_attn ops are not compatible 
 * with the PyTorch library bindings. To get around that we use 
 * `make_pytorch_shim` which creates a lambda that exponses the API using 
 * PyTorch compatible types to the types, then converts them to the types 
 * expected by the flash_attn ops. This shims allows us to make minimal changes
 * to `flash_api.cpp` making it easier to synchronize with upstream changes.
 * 
 * The `pytorch_library_compatible_type` struct is used to map from the 
 * flash_attn ops types to a PyTorch library compatible one. The main issues is
 * that the following types are not support by PyTorch libary bindings:
 *  - `int`
 *  - `float`
 *  - `c10::optional<T> &`
 *  - `c10::optional<const at::Tensor> &`
 * So we convert them to (respectively):
 *  - `int64_t`
 *  - `double`
 *  - `const c10::optional<T>&`
 *  - `const c10::optional<at::Tensor>&`
 */

template<typename T>
struct pytorch_library_compatible_type { 
    using type = T;
    static T convert_from_type(T arg) { return arg; }
};

// Map `c10::optional<T> &` -> `const c10::optional<T>&`
//  (NOTE: this is bit unsafe but non of the ops in flash_attn mutate 
//   the optional container)
template<typename T>
struct pytorch_library_compatible_type<c10::optional<T> &> { 
    using type = const c10::optional<T>&;
    static c10::optional<T>& convert_from_type(const c10::optional<T> &arg) { 
        return const_cast<c10::optional<T>&>(arg); 
    }
};

// Map `c10::optional<const at::Tensor>&` -> `const c10::optional<at::Tensor>&`
template<>
struct pytorch_library_compatible_type<c10::optional<const at::Tensor> &> { 
    using type = const c10::optional<at::Tensor>&;
    static c10::optional<const at::Tensor>& convert_from_type(
        const c10::optional<at::Tensor> &arg) {
        return const_cast<c10::optional<const at::Tensor>&>(
            reinterpret_cast<const c10::optional<const at::Tensor>&>(arg)); 
    }
};

// Map `int` -> `int64_t`
template<> struct pytorch_library_compatible_type<int> { 
    using type = int64_t; 
    static int convert_from_type(int64_t arg) {
        TORCH_CHECK(arg <= std::numeric_limits<int>::max(), 
            "int64_t value is too large to be converted to int");
        TORCH_CHECK(arg >= std::numeric_limits<int>::min(), 
            "int64_t value is too small to be converted to int");
        return arg; 
    }
};

// Map `float` -> `double`
template<> struct pytorch_library_compatible_type<float> { 
    using type = double; 
    static float convert_from_type(double arg) { 
        TORCH_CHECK(std::abs(arg) <= std::numeric_limits<float>::max(), 
            "double value is too large to be converted to float");
        return arg; 
    }
};

//
//  Shim Utils
//
template<typename T>
using pytorch_library_compatible_type_t = \
    typename pytorch_library_compatible_type<T>::type;

template<typename T>
T convert_from_pytorch_compatible_type(pytorch_library_compatible_type_t<T> arg) 
    { return pytorch_library_compatible_type<T>::convert_from_type(arg); }

template <typename Ret, typename... Args>
auto make_pytorch_shim(Ret(*fun)(Args... args)){
    return [fun](pytorch_library_compatible_type_t<Args>... args) {
        return fun(convert_from_pytorch_compatible_type<Args>(args)...);
    };
}

/**
 *  Torch Library Registration
 */
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("varlen_fwd(Tensor! q, Tensor k, Tensor v, Tensor!? out, Tensor cu_seqlens_q, "
            "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, "
            "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
            "bool is_causal, int window_size_left, int window_size_right, float softcap, bool return_softmax, "
            "Generator? gen) -> Tensor[]");
    ops.impl("varlen_fwd", torch::kCUDA, make_pytorch_shim(&mha_varlen_fwd));
    ops.impl("varlen_fwd", torch::kMeta, make_pytorch_shim(&mha_varlen_fwd_meta));

    ops.def("fwd_kvcache(Tensor! q, Tensor kcache, Tensor vcache, Tensor? k, Tensor? v, Tensor? seqlens_k, "
            "Tensor? rotary_cos, Tensor? rotary_sin, Tensor? cache_batch_idx, Tensor? leftpad_k, Tensor? block_table, "
            "Tensor? alibi_slopes, Tensor!? out, float softmax_scale, bool is_causal, int window_size_left, "
            "int window_size_right, float softcap, bool is_rotary_interleaved, int num_splits) -> Tensor[]");
    ops.impl("fwd_kvcache", torch::kCUDA, make_pytorch_shim(&mha_fwd_kvcache));
    ops.impl("fwd_kvcache", torch::kMeta, make_pytorch_shim(&mha_fwd_kvcache_meta));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME);