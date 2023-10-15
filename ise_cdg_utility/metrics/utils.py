import typing


def atomic_map(func: typing.Callable, l: typing.List) -> typing.List:
    def rec_func(el):
        if isinstance(el, list):
            return list(map(rec_func, el))
        return func(el)
    return list(map(rec_func, l))
