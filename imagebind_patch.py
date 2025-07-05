#!/usr/bin/env python
"""
Compatibility patch for ImageBind with newer PyTorch/torchvision versions.
"""

import sys
import torchvision.transforms.functional as F


class MockFunctionalTensor:
    """Mock module for backwards compatibility with functional_tensor imports."""

    def __getattr__(self, name):
        return getattr(F, name)


sys.modules["torchvision.transforms.functional_tensor"] = MockFunctionalTensor()

try:
    import pytorchvideo.transforms.functional

    if not hasattr(pytorchvideo.transforms.functional, "functional_tensor"):
        pytorchvideo.transforms.functional.functional_tensor = MockFunctionalTensor()
except ImportError:
    pass
