#!/usr/bin/env bash
set -euo pipefail

# Install CLI tools used in earlier repo setup.
npm install -g @anthropic-ai/claude-code
npm install -g @openai/codex

# Optional Anthropic-compatible endpoint configuration.
# Export ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN before running if needed.

# Install basic dependencies.
pip install numpy torch torchvision

# Install mamba-ssm precompiled wheel first.
pip install https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2+cu122torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Install remaining Python requirements.
pip install -r requirements.txt

IS_SANDBOX=1 claude --dangerously-skip-permissions