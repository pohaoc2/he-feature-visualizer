"""Compatibility wrapper for the relocated stage-1 cell-feature module."""

from tools.paper.figures.stage1 import vis_cell_feature_tiles as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


if __name__ == "__main__":
    main()
