# Leaf Vein Workflow

napari plugin for leaf vein image workflow development.

## MVP

- Open an image with the standard napari image reader.
- Calibrate pixel size from DPI, µm/px, mm/px, or one manually drawn scale-bar
  line; uncalibrated images remain usable in pixel units.
- Automatically create a rectangle ROI Shapes layer above the image.
- Automatically switch that Shapes layer to rectangle drawing mode.
- Add or update a cropped ROI Image layer only after clicking `Add ROI`.
- Create a smoothed Image layer only after clicking `Apply Gaussian`.
- Create an editable vein mask Labels layer only after clicking `Apply Otsu`.
- Clean the current mask with explicit morphology settings before manual edits.
- Create an editable order labels layer from the confirmed vein mask
  (1 primary, 2 secondary, 3 tertiary, 4 uncertain).
- Trace 1° and 2° veins as paths on the whole leaf (segmented-line method of
  Sack & Scoffoni 2013) and export their summed length and VLA.
- Measure skeleton length per vein order and VLA against a leaf area typed in
  mm² or cm², and export the results.

Saving cropped images, multiple ROIs, polygon ROIs, nD stacks, automatic vein
order classification, and minor-vein (4th order and above) labels are outside
the current MVP.

## Development

Install the project, development tools, and the local Qt backend:

```powershell
uv sync
```

`PyQt6` is included in the development dependency group, so local development
does not require a separate `uv pip install PyQt6` step.

Run napari:

```powershell
uv run napari
```

Then open the widget from:

```text
Plugins > Leaf Vein Workflow > Leaf Vein Workflow
```

With the widget open, use napari's normal image loading workflow. The plugin
creates and activates the ROI drawing layer automatically when an image layer is
added. Before or after creating the ROI, choose a scale calibration method and
click `Apply Scale`. For a visible scale bar, select `Scale bar`, click `Draw
Scale Bar Line`, draw exactly one line along the bar, enter its real length, and
then apply the scale. The normalized mm/px value is stored in layer metadata and
inherited by derived workflow layers; the plugin does not change napari layer
`scale` or `translate` transforms. The status shows µm/px, mm/px, and the
equivalent DPI regardless of the selected input method. Applying a scale also
draws a magenta display bar near the bottom-left of the image (10 mm when it
fits within 40% of the image width, otherwise 5, 2, 1 mm, or 500, 200, 100 µm)
so the calibration can be checked against the ruler by eye. The bar layer is
for display only and is updated, not duplicated, when the scale changes.

Draw one rectangle, then click `Add ROI` in the widget to create the ROI crop
layer. Click `Apply Gaussian` to create `smoothed_image`, then click
`Apply Otsu` to create `vein_mask`. After that explicit Otsu step, `Invert
threshold` and `Mask threshold` update the current mask.

If needed, set the morphology controls and click `Clean Mask`. Correct the mask
manually in paint mode, then click `Create Order Labels` to begin manual
vein-order annotation inside the confirmed mask. Cleanup inputs remain integer
pixel values for reproducible morphology behavior; when the image is calibrated,
their labels also show object/hole areas in mm² and closing/opening radii in µm
or mm.

In the `Measurement` section, type the leaf area (for example measured from a
scan taken before specimen preparation) and choose mm² or cm²; it is converted
to mm² internally. Click `Export Measurements` to measure the current labels and
write them to the output folder, which defaults to `output/` under the folder
napari was started from (`Change...` picks another one). The files are a
measurements CSV with total length and VLA (mm mm⁻²) for 1°, 2°, 3°, 1°+2°, and
major (1°–3°), the order labels as `.npy`, a QC overlay PNG, and a skeleton
branch table. Combined rows are the sum of the per-order skeleton lengths.
Results are not shown in the dock; a notification confirms the export and warns
about pixels still labelled 4 (uncertain), which are excluded from every order.

For 1° and 2° veins on a whole-leaf image, open `Vein paths (1°/2°)`, choose
the order, click `Draw Vein Paths`, and trace each vein (click vertices,
double-click to finish). New paths take the selected order; select existing
paths and change the order to relabel them. `Export Measurements` then also
writes `*_path_measurements.csv` (paths per order, length in px and mm, VLA for
1°, 2°, and 1°+2°) and `*_path_vertices.csv` (every traced vertex). Tertiary
veins are not traced here; measure them in sample boxes with order labels.

### Reproducibility metadata

Each export writes `<timestamp>_<image>_metadata.yaml` next to the results, and
every file from one export shares that prefix. It records the source file path
and SHA-256, calibration, leaf area as typed (value and unit) and in mm², the
parameters actually applied (ROI bounds, Gaussian sigma, Otsu and final
threshold, invert, every Clean Mask run), whether the mask was edited by hand
after the last automatic step, label pixel counts, number of traced paths, the
settings defaults in effect, and software versions. The vein mask itself is
saved as `*_vein_mask.npy` so hand edits are preserved.

### Settings

Default values (calibration method and DPI, display bar length, Gaussian sigma,
invert threshold, cleanup sizes, leaf-area unit, output folder) come from a YAML
file read when the widget opens. `leaf_vein_workflow.yaml` in the folder napari
is started from takes precedence; otherwise the per-user `config.yaml` is used
(its path is shown in the `Settings` section). `Open Settings File` creates a
commented file with the recommended values if none exists. Keys you leave out,
and invalid values, fall back to the recommended values with a warning.
`Reload` applies edits; `Reset to Defaults` rewrites the recommended file and
keeps the previous one as `*.bak`. The recommended scanner resolution is
1200 dpi, enough to resolve 2nd-order vein width.

Workflow sections open one at a time, and finishing a stage (Apply Scale, Add
ROI, Create Order Labels) opens the next one, so the dock rarely needs
scrolling.

Run tests:

```powershell
uv run pytest
```
