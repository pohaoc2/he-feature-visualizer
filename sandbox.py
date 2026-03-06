from skimage.registration import phase_cross_correlation
import numpy as np
import tifffile
he_path = "data/CRC02-HE.ome.tif"
multi_path = "data/CRC02.ome.tif"

# 1. Read level 5 or 6 (low res) from both files
with tifffile.TiffFile(he_path) as he, tifffile.TiffFile(multi_path) as multi:
    thumb_he = he.series[0].levels[-1].asarray()
    thumb_multi = multi.series[0].levels[-1].asarray()

# 2. Match scales and perform Phase Correlation
# (This gives you the pixel shift at that specific pyramid level)
shift, error, diffphase = phase_cross_correlation(thumb_he, thumb_multi)
print(f"Detected Offset at low-res: {shift}")
