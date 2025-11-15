"""Cross-platform emoji/symbol support."""

from typing import Dict


def get_symbols() -> Dict[str, str]:
    """Get appropriate symbols based on platform/terminal support.

    Returns:
        Dict[str, str]: Dictionary of symbol names to their display representations

    """
    try:
        # Test if terminal supports unicode
        test = "🔍"
        print(test, end="")
        print("\b \b", end="")  # backspace and clear
        return {
            "search": "🔍",
            "found": "📋",
            "success": "✅",
            "bye": "👋"
        }
    except UnicodeEncodeError:
        return {
            "search": "[Search]",
            "found": "[Found]",
            "success": "[Success]",
            "bye": "[Bye]",
        }


# Initialize symbols once
SYMBOLS = get_symbols()
