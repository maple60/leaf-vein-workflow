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
  Sack & Scoffoni 2013) and export their summed length and VLA. This is
  implemented but hidden from the dock for now.
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

The dock shows six numbered steps, one open at a time:

1. **Image & scale**: open the leaf scan and apply the scan resolution.
2. **ROI**: drag the rectangle over the area to analyze and press `Add ROI`.
3. **Vein mask**: `Apply Gaussian`, `Apply Otsu`, adjust the threshold, fix
   spots by hand, and optionally `Clean Mask`.
4. **Paint vein orders**: `Create Order Labels`, choose an order in
   `Paint as`, and drag over veins.
5. **Leaf area & VLA**: type the fresh-leaf area.
6. **Export**: choose the output folder and press `Export Measurements`.

Each step starts with a one-line instruction. Details follow.

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
`Apply Otsu` to create `vein_mask`. Colored (yellow) pixels are vein.
With `ROI includes background (paper)` on (the default), Otsu first separates
the leaf from the background around it, then finds the vein threshold inside
the leaf only, so a whole-leaf ROI works and the paper is never part of the
mask. If no background is found (an ROI inside the blade), it uses the whole
ROI. Otsu then picks the side with fewer pixels as vein and sets `Invert (dark
pixels are vein)` to match. The status line shows the share colored as vein
and warns when it is more than veins usually cover. After that explicit Otsu
step, `Invert` and `Mask threshold` update the current mask.

If needed, set the morphology controls and click `Clean Mask`. Correct the mask
manually in paint mode, then click `Create Order Labels` to begin manual
vein-order annotation inside the confirmed mask. Every vein pixel starts as
4 (uncertain, purple). Choose 1°, 2°, 3°, or Uncertain in `Paint as` (each
item shows its label color) and drag over veins; pixels outside the mask are
never painted. Picking a label in napari's own layer controls updates
`Paint as` too. Cleanup inputs remain integer
pixel values for reproducible morphology behavior; when the image is calibrated,
their labels also show object/hole areas in mm² and closing/opening radii in µm
or mm.

In `5. Leaf area & VLA`, type the leaf area (for example measured from a
scan taken before specimen preparation) and choose mm² or cm²; it is converted
to mm² internally. In `6. Export`, click `Export Measurements` to measure the current labels and
write them to the output folder, which defaults to `output/` under the folder
napari was started from (`Change...` picks another one). The files are a
measurements CSV with total length and VLA (mm mm⁻²) for 1°, 2°, 3°, 1°+2°, and
major (1°–3°), the order labels as `.npy`, a QC overlay PNG, and a skeleton
branch table. Combined rows are the sum of the per-order skeleton lengths.
Results are not shown in the dock; a notification confirms the export and warns
about pixels still labelled 4 (uncertain), which are excluded from every order.
Export runs in the background so napari stays usable; a full 3200 dpi leaf
takes about 30 s.

The display scale bar stays above layers added later, `vein_mask` is drawn in
yellow at 50% opacity, and `order_labels` at 60% opacity so the veins beneath
stay visible.

Path tracing for 1° and 2° veins is implemented but hidden from the dock for
now. When it is shown again: on a whole-leaf image, choose
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
file read when the widget opens. The repository ships
`leaf_vein_workflow.example.yaml` with the recommended values. When napari is
started from the repository folder and `leaf_vein_workflow.yaml` does not exist,
it is created from the example; edit that copy to customize (it is ignored by
git, so your values never conflict with updates). Copying the example over it
restores the recommended values. When napari is started from another folder,
the per-user `config.yaml` is used instead (its path is shown in the `Settings`
section). `Open Settings File` creates a commented file with the recommended
values if none exists. Keys you leave out,
and invalid values, fall back to the recommended values with a warning.
`Reload` applies edits; `Reset to Defaults` rewrites the recommended file and
keeps the previous one as `*.bak`. The recommended scanner resolution is
1200 dpi, enough to resolve 2nd-order vein width.

Workflow sections open one at a time, and finishing a stage (Apply Scale, Add
ROI) opens the next one, so the dock rarely needs scrolling. After `Create
Order Labels` the painting step stays open.

Run tests:

```powershell
uv run pytest
```
