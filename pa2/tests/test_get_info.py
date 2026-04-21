from typing import Dict
from mpi4py import MPI
import numpy as np
import pytest

from model.func_impl import get_info  # Updated get_info now accepts fc_layer

def check_info(
    input_dict: Dict,
    expect_output_dict: Dict,
) -> None:
    rank = input_dict["rank"]

    # Call the updated get_info with fc_layer string.
    mp_group_idx, dp_group_idx, mp_comm, dp_comm, part_in_dim, part_out_dim = get_info(
        comm=input_dict["comm"],
        rank=rank,
        mp_size=input_dict["mp_size"],
        dp_size=input_dict["dp_size"],
        fc_layer=input_dict["fc_layer"],
        in_dim=input_dict["in_dim"],
        out_dim=input_dict["out_dim"],
    )

    # Check communicator indices.
    assert mp_group_idx == expect_output_dict["mp_group_idx"][rank]
    assert dp_group_idx == expect_output_dict["dp_group_idx"][rank]
    assert part_in_dim == expect_output_dict["part_in_dim"]
    assert part_out_dim == expect_output_dict["part_out_dim"]

    local_arr = input_dict["input_array"][rank]

    mp_group_reduction_arr = np.empty_like(local_arr)
    dp_group_reduction_arr = np.empty_like(local_arr)

    # Allreduce over the model-parallel communicator (grouped by dp_idx).
    mp_comm.Allreduce(local_arr, mp_group_reduction_arr, op=MPI.SUM)
    # Allreduce over the data-parallel communicator (grouped by mp_idx).
    dp_comm.Allreduce(local_arr, dp_group_reduction_arr, op=MPI.SUM)

    np.testing.assert_allclose(
        actual=mp_group_reduction_arr,
        desired=expect_output_dict["mp_group_array"][dp_group_idx],
    )
    np.testing.assert_allclose(
        actual=dp_group_reduction_arr,
        desired=expect_output_dict["dp_group_array"][mp_group_idx],
    )


@pytest.mark.mpi
def test_fc_q():
    """
    Test for a fully-connected layer that partitions along the output dimension,
    e.g. 'fc_q' (as well as 'fc_k' or 'fc_v').
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Each rank gets a row from an 8x10 array.
    array = np.arange(80).reshape((8, 10))

    input_dict = {
        "comm": comm,
        "rank": rank,
        "mp_size": 4,   # Changed from 2 to 4.
        "dp_size": 2,   # Changed from 4 to 2.
        "fc_layer": "fc_q",
        "in_dim": 768,
        "out_dim": 256,
        "input_array": array,
    }

    # Compute expected communicator indices:
    # For mp: rank % 4; for dp: rank // 4.
    expect_output_dict = {
        "mp_group_idx": {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 0,
            5: 1,
            6: 2,
            7: 3,
        },
        "dp_group_idx": {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
        },
        # For 'fc_q': no partitioning on input; partition out_dim among mp groups.
        "part_in_dim": 768,
        "part_out_dim": 256 // 4,  # 64
        # Expected Allreduce for mp_comm (grouped by dp_idx):
        # dp group 0: ranks 0,1,2,3; dp group 1: ranks 4,5,6,7.
        "mp_group_array": {
            0: np.array([0+10+20+30, 1+11+21+31, 2+12+22+32, 3+13+23+33,
                         4+14+24+34, 5+15+25+35, 6+16+26+36, 7+17+27+37,
                         8+18+28+38, 9+19+29+39]),
            1: np.array([40+50+60+70, 41+51+61+71, 42+52+62+72, 43+53+63+73,
                         44+54+64+74, 45+55+65+75, 46+56+66+76, 47+57+67+77,
                         48+58+68+78, 49+59+69+79]),
        },
        # Expected Allreduce for dp_comm (grouped by mp_idx):
        # mp group 0: ranks 0 and 4; mp group 1: ranks 1 and 5; etc.
        "dp_group_array": {
            0: np.array([0+40, 1+41, 2+42, 3+43, 4+44, 5+45, 6+46, 7+47, 8+48, 9+49]),
            1: np.array([10+50, 11+51, 12+52, 13+53, 14+54, 15+55, 16+56, 17+57, 18+58, 19+59]),
            2: np.array([20+60, 21+61, 22+62, 23+63, 24+64, 25+65, 26+66, 27+67, 28+68, 29+69]),
            3: np.array([30+70, 31+71, 32+72, 33+73, 34+74, 35+75, 36+76, 37+77, 38+78, 39+79]),
        },
    }
    # For clarity, precompute the sums for the expected arrays.
    # mp_group_array for dp group 0:
    expect_output_dict["mp_group_array"][0] = np.array([
        0+10+20+30, 1+11+21+31, 2+12+22+32, 3+13+23+33, 4+14+24+34,
        5+15+25+35, 6+16+26+36, 7+17+27+37, 8+18+28+38, 9+19+29+39
    ])
    # mp_group_array for dp group 1:
    expect_output_dict["mp_group_array"][1] = np.array([
        40+50+60+70, 41+51+61+71, 42+52+62+72, 43+53+63+73, 44+54+64+74,
        45+55+65+75, 46+56+66+76, 47+57+67+77, 48+58+68+78, 49+59+69+79
    ])
    # dp_group_array for each mp group is already defined above.
    check_info(
        input_dict=input_dict,
        expect_output_dict=expect_output_dict,
    )


@pytest.mark.mpi
def test_fc_o():
    """
    Test for a fully-connected layer that partitions along the input dimension,
    e.g. 'fc_o'.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    array = np.arange(80).reshape((8, 10))

    input_dict = {
        "comm": comm,
        "rank": rank,
        "mp_size": 4,
        "dp_size": 2,
        "fc_layer": "fc_o",
        "in_dim": 256,
        "out_dim": 10,
        "input_array": array,
    }

    expect_output_dict = {
        "mp_group_idx": {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 0,
            5: 1,
            6: 2,
            7: 3,
        },
        "dp_group_idx": {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
        },
        # For 'fc_o': partition along the input dimension.
        "part_in_dim": 256 // 4,  # 64
        "part_out_dim": 10,
        "mp_group_array": {
            0: np.array([0+10+20+30, 1+11+21+31, 2+12+22+32, 3+13+23+33,
                         4+14+24+34, 5+15+25+35, 6+16+26+36, 7+17+27+37,
                         8+18+28+38, 9+19+29+39]),
            1: np.array([40+50+60+70, 41+51+61+71, 42+52+62+72, 43+53+63+73,
                         44+54+64+74, 45+55+65+75, 46+56+66+76, 47+57+67+77,
                         48+58+68+78, 49+59+69+79]),
        },
        "dp_group_array": {
            0: np.array([0+40, 1+41, 2+42, 3+43, 4+44, 5+45, 6+46, 7+47, 8+48, 9+49]),
            1: np.array([10+50, 11+51, 12+52, 13+53, 14+54, 15+55, 16+56, 17+57, 18+58, 19+59]),
            2: np.array([20+60, 21+61, 22+62, 23+63, 24+64, 25+65, 26+66, 27+67, 28+68, 29+69]),
            3: np.array([30+70, 31+71, 32+72, 33+73, 34+74, 35+75, 36+76, 37+77, 38+78, 39+79]),
        },
    }
    # Precompute the expected arrays for clarity.
    expect_output_dict["mp_group_array"][0] = np.array([
        0+10+20+30, 1+11+21+31, 2+12+22+32, 3+13+23+33, 4+14+24+34,
        5+15+25+35, 6+16+26+36, 7+17+27+37, 8+18+28+38, 9+19+29+39
    ])
    expect_output_dict["mp_group_array"][1] = np.array([
        40+50+60+70, 41+51+61+71, 42+52+62+72, 43+53+63+73, 44+54+64+74,
        45+55+65+75, 46+56+66+76, 47+57+67+77, 48+58+68+78, 49+59+69+79
    ])
    check_info(
        input_dict=input_dict,
        expect_output_dict=expect_output_dict,
    )
