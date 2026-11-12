#!/usr/bin/env python3
"""Report only CUDA devices visible to this Python environment as JSON.

Run in a short-lived subprocess so CUDA query contexts are released afterward.
No model is loaded. GPU utilization is optional (PyTorch/NVML support required).
"""
import json
import os
from datetime import datetime, timezone


def error_details(error):
    """Name missing dependencies without mislabelling driver failures."""
    if isinstance(error, ImportError):
        name = getattr(error, 'name', None) or ''
        message = str(error).lower()
        if name == 'pynvml' or 'pynvml' in message or 'nvidia-ml-py' in message:
            return ('nvidia-ml-py is not available or could not be imported. '
                    'Install it in the UI Python environment: '
                    'python -m pip install nvidia-ml-py', 'nvidia-ml-py')
        if name == 'torch':
            return ('PyTorch (torch) is not available in the UI Python environment.', 'torch')
    return str(error), None


def record_dependency(result, package):
    if package and package not in result['missing_packages']:
        result['missing_packages'].append(package)


def collect(torch):
    result = {'checked_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'cuda_available':bool(torch.cuda.is_available()), 'devices':[], 'error':None,
              'warnings':[], 'missing_packages':[]}
    if not result['cuda_available']:
        return result
    for index in range(torch.cuda.device_count()):
        device = {'device':f'cuda:{index}', 'name':None, 'free_bytes':None,
                  'total_bytes':None, 'utilization_percent':None,
                  'utilization_error':None, 'error':None}
        try:
            device['name'] = torch.cuda.get_device_name(index)
            free,total = torch.cuda.mem_get_info(index)
            device.update(free_bytes=int(free), total_bytes=int(total))
        except Exception as error:
            device['error'], package = error_details(error)
            record_dependency(result, package)
        try:
            # May require an optional NVML package; memory queries do not.
            device['utilization_percent'] = int(torch.cuda.utilization(index))
        except Exception as error:
            message, package = error_details(error)
            record_dependency(result, package)
            device['utilization_error'] = message
            warning = 'GPU utilization is unavailable: ' + message
            if warning not in result['warnings']:
                result['warnings'].append(warning)
        result['devices'].append(device)
    return result


def main():
    try:
        import torch
        result = collect(torch)
    except Exception as error:
        message, package = error_details(error)
        result = {'checked_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'cuda_available':None, 'devices':[], 'error':message,
                  'warnings':[], 'missing_packages':[package] if package else []}
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__':
    main()
