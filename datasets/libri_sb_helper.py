import torch


def libri_padding(raw_batch):
    batch = {}
    for key in ['audio', 'label', 'label_bos', 'label_eos']:
        data = [b[key] for b in raw_batch]
        data, length = batch_pad_right(data)
        batch[key] = data  
        batch[key+'_length'] = length
    task_id = [b['task_id'] for b in raw_batch]
    batch['task_id'] = torch.cat(task_id).unsqueeze(-1)
    return batch
    


def pad_right_to(
    tensor: torch.Tensor, target_shape: (list, tuple), mode="constant", value=0,
):
    assert len(target_shape) == tensor.ndim
    pads = []  # this contains the abs length of the padding for each dimension.
    valid_vals = []  # this contains the relative lengths for each dimension.
    i = len(target_shape) - 1  # iterating over target_shape ndims
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        valid_vals.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1

    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid_vals


def batch_pad_right(tensors: list, mode="constant", value=0):
    max_shape = []
    for dim in range(tensors[0].ndim):
        if dim != 0:
            if not all(
                [x.shape[dim] == tensors[0].shape[dim] for x in tensors[1:]]
            ):
                raise EnvironmentError(
                    "Tensors should have same dimensions except for the first one"
                )
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        # for each tensor we apply pad_right_to
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value
        )
        batched.append(padded)
        valid.append(valid_percent[0])

    batched = torch.stack(batched)

    return batched, torch.tensor(valid)