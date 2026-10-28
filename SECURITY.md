# Security Policy

## Scope

AttentionVision is an offline/local academic machine learning research repository for image classification experiments on CIFAR-10. It does not expose web servers, external APIs, or authentication systems.

## Safe Checkpoint Loading and Deserialization

PyTorch checkpoints (`.pt` / `.pth`) saved via `torch.save` rely on Python's `pickle` protocol when saving arbitrary dictionaries. Loading untrusted `.pt` files from unknown third-party sources can execute arbitrary code.

- **Best Practice**: Only load checkpoints that you generated locally or verified via cryptographic checksums.
- **Implementation**: `src/attentionvision/train.py` validates checkpoint paths and uses `weights_only=False` only for structured local experiment checkpoints containing configuration dictionaries and optimizer states, with fallback support and explicit structure validation.

## Reporting Security Issues

If you discover a vulnerability (such as unsafe path traversal in configuration loading or dependency vulnerabilities):
1. Do not open a public issue containing exploit payloads.
2. Document the affected file, reproduction steps, and proposed fix in a private security advisory on GitHub.
