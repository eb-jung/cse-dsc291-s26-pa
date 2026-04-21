from typing import Dict, List

import numpy as np
import pytest

from data.data_parallel_preprocess import split_data


def check_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    mp_size: int,
    dp_size: int,
    rank: int,
    expected_x_train_ret: np.ndarray,
    expected_y_train_ret: np.ndarray,
) -> None:
    x_train_ret, y_train_ret = split_data(
        x_train=x_train,
        y_train=y_train,
        mp_size=mp_size,
        dp_size=dp_size,
        rank=rank,
    )

    # Check that the total number of samples (when reassembled) is preserved.
    assert (
        x_train_ret.shape[0] * dp_size == x_train.shape[0]
    ), f"x_train shape mismatch should be {expected_x_train_ret.shape}"
    assert (
        y_train_ret.shape[0] * dp_size == y_train.shape[0]
    ), f"y_train shape mismatch should be {expected_y_train_ret.shape}"

    np.testing.assert_allclose(actual=x_train_ret, desired=expected_x_train_ret)
    np.testing.assert_allclose(actual=y_train_ret, desired=expected_y_train_ret)


# In these tests, each sample is now a 2x2 matrix.
# We define 8 samples for x_train.
x_train_full = np.array([
    [[ 1.0,  2.0],
     [ 3.0,  4.0]],
    [[ 5.0,  6.0],
     [ 7.0,  8.0]],
    [[ 9.0, 10.0],
     [11.0, 12.0]],
    [[13.0, 14.0],
     [15.0, 16.0]],
    [[17.0, 18.0],
     [19.0, 20.0]],
    [[21.0, 22.0],
     [23.0, 24.0]],
    [[25.0, 26.0],
     [27.0, 28.0]],
    [[29.0, 30.0],
     [31.0, 32.0]],
])
# And for y_train, each sample is now a 2-element vector.
y_train_full = np.array([
    [ 1.0,  2.0],
    [ 3.0,  4.0],
    [ 5.0,  6.0],
    [ 7.0,  8.0],
    [ 9.0, 10.0],
    [11.0, 12.0],
    [13.0, 14.0],
    [15.0, 16.0],
])


def test_mp_2_dp_1():
    # mp_size=2, dp_size=1 means each model-parallel rank gets the full dataset.
    mp_size = 2
    dp_size = 1

    rank_to_x_train = {
        0: x_train_full,
        1: x_train_full,
    }

    rank_to_y_train = {
        0: y_train_full,
        1: y_train_full,
    }

    for rank in rank_to_x_train:
        check_split(
            x_train_full,
            y_train_full,
            mp_size,
            dp_size,
            rank,
            rank_to_x_train[rank],
            rank_to_y_train[rank],
        )


def test_mp_1_dp_2():
    # mp_size=1, dp_size=2 means the full set is split into 2 halves.
    mp_size = 1
    dp_size = 2
    # Split the 8 samples into 2 groups of 4.
    rank_to_x_train = {
        0: x_train_full[:4],
        1: x_train_full[4:],
    }

    rank_to_y_train = {
        0: y_train_full[:4],
        1: y_train_full[4:],
    }

    for rank in rank_to_x_train:
        check_split(
            x_train_full,
            y_train_full,
            mp_size,
            dp_size,
            rank,
            rank_to_x_train[rank],
            rank_to_y_train[rank],
        )


def test_mp_2_dp_2():
    # mp_size=2, dp_size=2 means:
    #   - First half (samples 0-3) is used for one dp split,
    #     and each model-parallel rank gets that same half.
    #   - Second half (samples 4-7) is used for the other dp split.
    mp_size = 2
    dp_size = 2
    # Prepare the two halves:
    first_half_x = x_train_full[:4]
    second_half_x = x_train_full[4:]
    first_half_y = y_train_full[:4]
    second_half_y = y_train_full[4:]

    rank_to_x_train = {
        0: first_half_x,
        1: first_half_x,
        2: second_half_x,
        3: second_half_x,
    }

    rank_to_y_train = {
        0: first_half_y,
        1: first_half_y,
        2: second_half_y,
        3: second_half_y,
    }

    for rank in rank_to_x_train:
        check_split(
            x_train_full,
            y_train_full,
            mp_size,
            dp_size,
            rank,
            rank_to_x_train[rank],
            rank_to_y_train[rank],
        )


def test_mp_2_dp_4():
    # mp_size=2, dp_size=4 means we split 8 samples into 4 groups of 2.
    mp_size = 2
    dp_size = 4
    # Divide the 8 samples into 4 groups:
    group0_x = x_train_full[0:2]
    group1_x = x_train_full[2:4]
    group2_x = x_train_full[4:6]
    group3_x = x_train_full[6:8]
    group0_y = y_train_full[0:2]
    group1_y = y_train_full[2:4]
    group2_y = y_train_full[4:6]
    group3_y = y_train_full[6:8]

    rank_to_x_train = {
        0: group0_x,
        1: group0_x,
        2: group1_x,
        3: group1_x,
        4: group2_x,
        5: group2_x,
        6: group3_x,
        7: group3_x,
    }

    rank_to_y_train = {
        0: group0_y,
        1: group0_y,
        2: group1_y,
        3: group1_y,
        4: group2_y,
        5: group2_y,
        6: group3_y,
        7: group3_y,
    }

    for rank in rank_to_x_train:
        check_split(
            x_train_full,
            y_train_full,
            mp_size,
            dp_size,
            rank,
            rank_to_x_train[rank],
            rank_to_y_train[rank],
        )


if __name__ == "__main__":
    test_mp_2_dp_1()
    test_mp_1_dp_2()
    test_mp_2_dp_2()
    test_mp_2_dp_4()
