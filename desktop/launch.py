import runpy
import sys


def main() -> None:
    # Desktop launcher. Equivalent to: python -m desktop.main_qt
    try:
        runpy.run_module("desktop.main_qt", run_name="__main__")
    except ModuleNotFoundError as e:
        print("Module not found. Reinstall project: pip install -e .")
        print("Error:", e)
        sys.exit(1)
    except Exception as e:
        print("Desktop launch error:", e)
        raise


if __name__ == "__main__":
    main()
