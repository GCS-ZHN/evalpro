"""
Utility functions for xevaluate
"""

__all__ = [
    'get_docstring_description',
]

def get_docstring_description(cls):
    """Get the first paragraph of the class docstring."""
    if cls.__doc__ is None:
        return None

    content = ''
    for line in cls.__doc__.split('\n'):
        line = line.strip()
        if not line:
            if content:
                break
        else:
            if content:
                line = ' ' + line
            content += line
    return content
