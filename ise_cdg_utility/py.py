def each(fn, iterable, keep_results: bool = True):
    results = None
    for item in iterable:
        result = fn(item)
        if keep_results:
            results = (results or []) + [result]
    return results
