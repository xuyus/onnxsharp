import onnx, os


def enforce(status, msg):
    if status is not True:
        raise RuntimeError("exception raised during execution: ", msg)


class Type(object):
    def __init__(self) -> None:
        pass


def save_onnx_model_to_string(model_proto, path):
    if os.path.exists(path):
        print(f"WARNING: File {path} already exists, will be overwritten.")

    text_file = open(path, "w")
    text_file.write(str(model_proto))
    text_file.close()
    print(f"Saved model serialized string to {path} successfully.")


def save_onnx_model(
    model_proto,
    path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="filename",
    size_threshold=1024,
    convert_attribute=False,
):

    if os.path.exists(path):
        print(
            f"WARNING: {path} already exists, to make sure the written file is 100% usable,"
            " suggest you to remove it manually."
        )

    onnx.save_model(
        model_proto,
        path,
        save_as_external_data,
        all_tensors_to_one_file,
        location,
        size_threshold,
        convert_attribute,
    )
    print(
        f"Saved model to {path} successfully."
        + (f"External data location: {location}" if save_as_external_data else "")
    )
