from typing import Dict, List
from mpi4py import MPI

import numpy as np
import pytest

from model.func_impl import naive_collect_backward_output, naive_collect_backward_x


def check_naive_mp_backward_output(
    input_dict: Dict,
    expect_output_dict: Dict,
) -> None:
    x = input_dict["input_x"]

    output = naive_collect_backward_output(
        output_grad=x,
        mp_size=input_dict["mp_size"],
        mp_group_idx=input_dict["mp_group_idx"],
    )

    assert x.dtype == output.dtype

    np.testing.assert_allclose(
        actual=output, desired=expect_output_dict["output_array"]
    )


def check_naive_mp_backward_x(
    input_dict: Dict,
    expect_output_dict: Dict,
) -> None:
    x = input_dict["input_x"]

    output = naive_collect_backward_x(
        grad_x=x,
        mp_size=input_dict["mp_size"],
        mp_comm=input_dict["mp_comm"],
    )

    assert x.dtype == output.dtype

    np.testing.assert_allclose(
        actual=output, desired=expect_output_dict["output_array"]
    )


# ========================= 3-D Tests =========================

# For backward_output, we assume a 3-D tensor of shape
#   (batch_size, seq_length, out_dim)
# where the fc layer’s weight is partitioned along the last dimension.
# For example, if batch_size=1, seq_length=4, out_dim=8, and mp_size=4,
# then each rank should receive a tensor of shape (1, 4, 2) corresponding to a
# contiguous slice along the last dimension.
@pytest.mark.mpi
def test_fc2_naive_mp_backward_output_3d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mp_size = 4

    # Create a 3-D tensor: shape (batch_size, seq_length, out_dim)
    # Let batch_size = 1, seq_length = 4, out_dim = 8.
    array = np.arange(1 * 4 * 8).reshape((1, 4, 8)).astype(np.float64)

    # Expected: split along the last axis into 4 equal parts.
    # For example, rank 0 gets array[:, :, 0:2],
    # rank 1 gets array[:, :, 2:4], etc.
    output_array_list = {
        0: array[:, :, 0:2],
        1: array[:, :, 2:4],
        2: array[:, :, 4:6],
        3: array[:, :, 6:8],
    }

    input_dict = {
        "input_x": array,
        "mp_group_idx": rank,
        "mp_size": mp_size,
    }
    expect_output_dict = {"output_array": output_array_list[rank]}

    check_naive_mp_backward_output(input_dict, expect_output_dict)


# For backward_x, we simulate a reduce-scatter.
# Suppose each rank’s local grad_x has shape
#   (batch_size, seq_length, in_dim)
# and we let in_dim = 8, batch_size = 1, seq_length = 3.
# We create a “global” tensor of shape (mp_size, 3, 8) and let each rank’s input be
# its own slice along the first dimension. Then the expected reduce-scatter result is
# computed by summing over the first dimension (across all ranks) and splitting the result
# along the last axis into 4 equal blocks.
@pytest.mark.mpi
def test_fc2_naive_mp_backward_x_3d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mp_size = 4

    # Create a global 3-D tensor of shape (mp_size, seq_length, in_dim)
    # Let batch_size (per rank) = 1, seq_length = 3, in_dim = 8.
    global_array = np.arange(mp_size * 3 * 8).reshape((mp_size, 3, 8)).astype(np.float64)

    # Each rank gets its slice along the first axis: shape (1, 3, 8)
    local_grad_x = global_array[rank : rank + 1]

    # Compute the global sum over the batch dimension (i.e. across all ranks)
    # This is what a reduce would do. The result has shape (1, 3, 8).
    global_sum = np.sum(global_array, axis=0, keepdims=True)

    # Then, reduce-scatter partitions global_sum along the last axis into mp_size blocks.
    # Here, each block has width in_dim // mp_size = 8 // 4 = 2.
    # Expected for rank r is: global_sum[:, :, r*2:(r+1)*2]
    output_array_list = {
        0: global_sum[:, :, 0:2],
        1: global_sum[:, :, 2:4],
        2: global_sum[:, :, 4:6],
        3: global_sum[:, :, 6:8],
    }

    input_dict = {
        "input_x": local_grad_x,
        "mp_comm": comm,
        "mp_size": mp_size,
    }
    expect_output_dict = {"output_array": output_array_list[rank]}

    check_naive_mp_backward_x(input_dict, expect_output_dict)
