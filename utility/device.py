def to_device(device, *tensors):
    in_device_tensors = []
    for tens in tensors:
        in_device_tensors.append(tens.to(device=device))
    return tuple(in_device_tensors)