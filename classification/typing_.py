from torch import Tensor


ImageBatch = Tensor
"""Image batch with shape (batch_size, 3, width, height)
"""

Logits = Tensor
"""Logits with shape (batch_size, num_classes)
"""

Targets = Tensor
"""Image classification targets (batch_size)
"""