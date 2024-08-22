from torch._subclasses.fake_tensor import _is_tensor_constructor
from torch.utils._python_dispatch import TorchDispatchMode


class MetaTensorContext(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if _is_tensor_constructor(func):
            device_idx = [arg.name
                          for arg in func._schema.arguments].index('device')
            if len(args) > device_idx:
                args = list(args)
                args[device_idx] = 'meta'
            else:
                kwargs['device'] = 'meta'
        return func(*args, **kwargs)
