KERNEL_CONFIGS = [
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "num_warps": 8, "num_stages": 4},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 4},
    {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "num_warps": 8, "num_stages": 4},
]


@triton.jit
def matmul_add_relu_kernel_fp16(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_dm,
    stride_dn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # Step 1: Tile: Assignment
    #
    # Each kernel instance is mapped to a tile in the output matrix C.
    # Compute the starting indices (m_start, n_start) for this tile.
    # -------------------------------------------------------------------------
    # TODO: Compute the tile indices using program_id(0) for M and program_id(1) for N.
    ...

    # -------------------------------------------------------------------------
    # Step 2: Register Tiling
    # -------------------------------------------------------------------------
    # TODO: Initialize the accumulator "acc" with zeros (dtype: float16 or float32).
    acc = ...

    # -------------------------------------------------------------------------
    # Step 3: Shared Memory Tiling & Cooperative Fetching.
    # Compute pointers to the sub-tiles of A and B that are needed to compute
    # the current C tile. The offsets here serve to load BLOCK_M x BLOCK_K
    # and BLOCK_K x BLOCK_N blocks from A and B respectively.
    # -------------------------------------------------------------------------
    # TODO: Finish code below.
    for k in range(0, ...):
        ...

    # -------------------------------------------------------------------------
    # Step 4: Add C and Apply ReLU to the accumulator
    # -------------------------------------------------------------------------
    # TODO: Finish code below.
    ...

    # -------------------------------------------------------------------------
    # Step 5: Write Cache / Epilogue Fusion: Write the computed tile to D.
    # -------------------------------------------------------------------------
    # TODO: Finish code below.
    ...
