def enforce(status, msg):
    if status is not True:
        raise RuntimeError("exception raised during execution: ", msg)


class Type(object):
    def __init__(self) -> None:
        pass
