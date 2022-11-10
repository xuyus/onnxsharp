import numpy

def npy_summurize_array(array, name=""):
    flatten_array = array.flatten()
    print(f"{name} shape: {array.shape} dtype: {array.dtype} size: {flatten_array.size} \n"
            f"min: {flatten_array.min()} max: {flatten_array.max()}, mean: {flatten_array.mean()}, std: {flatten_array.std()} \n"
            f"nan: {numpy.isnan(flatten_array).sum()}, inf: {numpy.isinf(flatten_array).sum()},")
    print(f"neg: {numpy.less(flatten_array, 0).sum()}, pos: {numpy.greater(flatten_array, 0).sum()}, zero: {numpy.equal(flatten_array, 0).sum()}, \n"
            f"norm: {numpy.linalg.norm(flatten_array)}, l2: {numpy.linalg.norm(flatten_array, ord=2)}, \n"
            f"histogram: {numpy.histogram(flatten_array, bins=max(1, flatten_array.size.bit_length() - 1)) if flatten_array.size > 2 else None} \n"
            f"==================================================================")
    #f"numpy.nonzero(array): {numpy.nonzero(flatten_array)}, \n"
    return numpy.isnan(flatten_array).sum() > 0 or numpy.isinf(flatten_array).sum() > 0