"""Shared predicates over the `data_mode` string.

`aligned_spatial` is always stacked as [z_ccf, y_ccf, x_ccf] (see
data_loader._align_and_tokenize_slices), so which column is the *section* axis
depends on how the brain was cut:

  * coronal datasets (Zhuang-ABCA-1 / -2, and the rq1 M550 sections) section
    along x_ccf -> section axis is column -1, in-plane coords are [:, :2] = (z, y)
  * sagittal datasets (Zhuang-ABCA-3, the rq4 family) section along z_ccf
    -> section axis is column 0, in-plane coords are [:, 1:] = (y, x)

This rule used to be spelled out as a literal `data_mode == "rq4"` at six call
sites, which silently excluded the other modes that also read Zhuang-ABCA-3
(`rq4_rq3`, `rq4_noref`). Keep it in one place.
"""


def is_sagittal_mode(data_mode) -> bool:
    """True for the rq4 family (Zhuang-ABCA-3, sectioned sagittally)."""
    return bool(data_mode) and str(data_mode).startswith("rq4")


def section_axis(data_mode) -> int:
    """Column of `aligned_spatial` that separates sections in this mode."""
    return 0 if is_sagittal_mode(data_mode) else -1
