__version__ = "2.7.2.post1"

# Use relative import to support build-from-source installation in vLLM
from .flash_attn_interface import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    is_fa2_supported,
    is_fa3_supported
)