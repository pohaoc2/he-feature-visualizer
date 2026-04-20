"""Compatibility wrapper for the relocated stage-1 overview module."""

from tools.paper.figures.stage1 import vis_he_mx_side_by_side as _impl

for _name in dir(_impl):
	if not _name.startswith("__"):
		globals()[_name] = getattr(_impl, _name)


if __name__ == "__main__":
	main()
