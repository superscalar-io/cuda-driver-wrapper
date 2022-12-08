
typedef unsigned __int128 u128;

static_assert(sizeof(ulong) == 8);
static_assert(sizeof(ulong2) == 16);
static_assert(sizeof(ulong4) == 32);
static_assert(sizeof(u128) == 16);

inline __device__ ulong2 adc(ulong a, ulong b, ulong carry)
{
    const u128 temp_a = a;
    const u128 temp_b = b;
    const u128 temp_carry = carry;

    const u128 result = temp_a + temp_b + temp_carry;

    return make_ulong2((ulong)result, (ulong)(result >> 64));
}

__device__ ulong4 add(const ulong4 a, const ulong4 b)
{
    const ulong2 temp0 = adc(a.x, b.x, 0);
    const ulong2 temp1 = adc(a.y, b.y, temp0.y);
    const ulong2 temp2 = adc(a.z, b.z, temp1.y);
    const ulong2 temp3 = adc(a.w, b.w, temp2.y);

    return make_ulong4(temp0.x, temp1.x, temp2.x, temp3.x);
}

#define DEC_IDX                                                  \
    int blk_i = (blockIdx.z * gridDim.y * gridDim.x) +           \
                (blockIdx.y * gridDim.x) +                       \
                blockIdx.x;                                      \
    int thd_i = (blk_i * blockDim.z * blockDim.y * blockDim.x) + \
                (threadIdx.z * blockDim.y * blockDim.x) +        \
                (threadIdx.y * blockDim.x) + threadIdx.x;

extern "C" __global__ void add_test(
    const ulong4 *A_in,
    const ulong4 *B_in,
    ulong4 *Out)
{
    DEC_IDX

    Out[thd_i] = add(A_in[thd_i], B_in[thd_i]);
}

extern "C" __global__ void add_test_2D_array_param(
    const ulong4 *AB_in,
    const size_t array_dim_x_size,
    ulong4 *Out)
{
    DEC_IDX

    const ulong4 a = AB_in[thd_i];
    const ulong4 b = AB_in[array_dim_x_size + thd_i];

    Out[thd_i] = add(a, b);
}