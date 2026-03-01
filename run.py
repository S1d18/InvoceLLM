#!/usr/bin/env python
"""
Entry point для Invoice LLM.

Использование:
    python run.py classify invoice.pdf
    python run.py batch M:/incoming --output results.csv
    python run.py status
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в path
root = Path(__file__).parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

if __name__ == '__main__':
    # GUI shortcut: python run.py gui
    if len(sys.argv) > 1 and sys.argv[1] == 'gui':
        from gui.app import main as gui_main
        gui_main()
        sys.exit(0)

    from cli.main import main
    sys.exit(main())
