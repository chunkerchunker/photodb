# Pre-import torchvision to avoid "kernel already registered" errors
# when tests import torch and torchvision in different test files.
# See: https://github.com/pytorch/vision/issues/7178
try:
    import torchvision  # noqa: F401
except ImportError:
    pass
