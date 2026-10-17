"""
Package init for memory module.
Allows: from memory.memory_manager import save_memory, retrieve_memory
"""

# Use relative imports within the package
from .memory_manager import save_memory, retrieve_memory, search_semantic, search_structured

try:
    from .memory_tools import search_personal_memory, save_personal_memory
except ImportError:
    # Fallback for direct execution
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from memory.memory_tools import search_personal_memory, save_personal_memory
