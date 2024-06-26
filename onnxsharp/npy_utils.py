# Copyright 2024 XUYUS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy


def npy_summurize_array(array, name=""):
    if array is None or not isinstance(array, numpy.ndarray):
        print(f"{name} not a numpy array, value: {array}")
        return None
    numpy.set_printoptions(
        suppress=True, precision=6, sign=" ", linewidth=128, floatmode="fixed"
    )
    flatten_array = array.flatten()

    num_nan = numpy.isnan(flatten_array).sum()
    num_inf = numpy.isinf(flatten_array).sum()
    print(
        f"{name} shape: {array.shape} dtype: {array.dtype} size: {flatten_array.size} \n"
        f"min: {flatten_array.min()} max: {flatten_array.max()}, mean: {flatten_array.mean()}, std: {flatten_array.std()} \n"
        f"nan: {num_nan}, inf: {num_inf}"
    )
    print(f"samples(top 128): {flatten_array[:128]}")

    print(
        f"neg: {numpy.less(flatten_array, 0).sum()}, pos: {numpy.greater(flatten_array, 0).sum()}, zero: {numpy.equal(flatten_array, 0).sum()},"
    )
    if num_nan + num_inf == 0:
        print(
            f"norm: {numpy.linalg.norm(flatten_array)}, l2: {numpy.linalg.norm(flatten_array, ord=2)}, \n"
            f"histogram: {numpy.histogram(flatten_array, bins=max(1, flatten_array.size.bit_length() - 1)) if flatten_array.size > 2 else None} \n"
            f"=================================================================="
        )
    # f"numpy.nonzero(array): {numpy.nonzero(flatten_array)}, \n"
    return numpy.isnan(flatten_array).sum() > 0 or numpy.isinf(flatten_array).sum() > 0
