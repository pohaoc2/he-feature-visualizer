"""Compatibility wrapper for the relocated stage-1 zoom figure module."""

from tools.paper.figures.stage1 import select_marker_patch_examples as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


if __name__ == "__main__":
    main()
