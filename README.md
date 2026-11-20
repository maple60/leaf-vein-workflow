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
- Review and edit `leaf_mask` after `Apply Otsu`; `Confirm Leaf & Extract Veins`
  measures its area and creates the vein mask inside that region.
- Clean the current mask with explicit morphology settings before manual edits.
- Create an editable order labels layer from the confirmed vein mask
  (1 primary, 2 secondary, 3 tertiary, 4 uncertain).
- Trace 1° and 2° veins as paths on the whole leaf (segmented-line method of
  Sack & Scoffoni 2013) and export their summed length and VLA. This is
  implemented but hidden from the dock for now.
- Measure per-order skeleton length and VLA using the confirmed specimen area
  in the same ROI, or an explicitly selected external area in mm² or cm².

Saving cropped images, multiple ROIs, polygon ROIs, nD stacks, automatic 3°
classification, and minor-vein (4th order and above) labels are outside the
current MVP. 1° and 2° veins can be auto-assigned (see below).

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
3. **Vein mask**: `Apply Gaussian`, `Apply Otsu`, edit the blade in `leaf_mask`,
   then `Confirm Leaf & Extract Veins`. Adjust the vein threshold, paint/erase,
   and optionally `Clean Mask`.
4. **Paint vein orders**: `Create Order Labels`, optionally mark the base and
   apex and auto-assign 1°/2°, then choose an order in `Paint as` and drag over
   veins to finish.
5. **Leaf area & VLA**: use the confirmed ROI mask, or select an external area.
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
`Apply Otsu` to create an editable `leaf_mask`. With background exclusion on,
the first Otsu split proposes the leaf region. Paint or erase to correct its
boundary, remove the petiole and debris, and decide how to handle real holes.
If no background is found (or exclusion is off), the full ROI is proposed and
must still be reviewed; its area is not automatically the whole-leaf area.

Click `Confirm Leaf & Extract Veins` to freeze that region, calculate its area
(px² × mm/px squared), and run a second Otsu inside it. Colored (yellow) pixels
in `vein_mask` are vein. The minority intensity class sets `Invert` automatically;
`Invert` and `Mask threshold` then update that vein mask. The confirmed leaf
mask is hidden to make vein editing easier; select and show it to edit again.
After leaf edits, reconfirm before thresholding or export. Re-extraction asks
before replacing the vein mask and removing its order labels and manual edits.

If needed, set the morphology controls and click `Clean Mask`. Correct the mask
manually in paint mode, then click `Create Order Labels` to begin manual
vein-order annotation inside the confirmed mask. Every vein pixel starts as
4 (uncertain, purple). Choose 1°, 2°, 3°, or Uncertain in `Paint as` (each
item shows its label color) and drag over veins; pixels outside the mask are
never painted. Picking a label in napari's own layer controls updates
`Paint as` too.

**Auto-assign 1°/2° (optional).** Click `Mark Base & Apex`, click the leaf base
(where the 1° enters the blade) and then the apex, and press `Auto-assign
1°/2°`. After the second click the dock switches back to painting on
`order_labels` (set `auto_order.activate_labels_after_marking: false` to stay on
the points). It runs in the background with a progress bar (about a minute for
a full leaf at 3200 dpi) and only changes pixels that are still Uncertain, so
anything painted by hand is kept. Export also shows a progress bar. The rules follow the Manual of Leaf Architecture (Ellis et al., 2009),
implemented in `auto_order.py`:

- 1°: the cheapest skeleton path from base to apex, with step cost
  length / diameter², so it keeps to the widest vein.
- 2° origins: each vein leaving the 1° is measured in a ring just outside the
  1° edge (junctions swell the width). It is a 2° candidate when it is at least
  70% of the thickest branch nearby on the same side, because secondaries thin
  toward the apex.
- 2° course: the cheapest path to the margin on the same side, cut where the
  gauge drops abruptly (a detour into a thinner vein).
- A candidate whose course mostly runs along a thicker 2° is a 3° joining the
  1° and a 2°, so it is not a 2°.
- 3° guess: the remaining vein segments are split by gauge. Segments at least
  as thick as a cutoff but thinner than the median 2° gauge become 3°; thinner
  ones (minor veins) stay Uncertain, and so do veins at both ends of the 1°
  (petiole, apex). The cutoff comes from Otsu on the length-weighted log gauge
  unless `auto_order.tertiary_min_gauge_mm` is set; `auto_order.assign_tertiary:
  false` leaves all remaining veins Uncertain. Labelling every remaining vein 3°
  is avoided on purpose: at 3200 dpi the mask also holds the minor-vein mesh,
  which would otherwise inflate VLA3.

On a 3200 dpi konara scan the 1° and about 30 secondaries are found, and the
3° guess is mostly the percurrent veins between secondaries (cutoff ≈ 6.4 px ≈
51 µm; the automatic cutoff is sensitive because the gauge distribution has no
clear valley, so check it). Veins near the leaf base still need hand correction. The clicks, parameters, and counts
are written to the export metadata under `order_labels.auto_order`. Cleanup inputs remain integer
pixel values for reproducible morphology behavior; when the image is calibrated,
their labels also show object/hole areas in mm² and closing/opening radii in µm
or mm.

In `5. Leaf area & VLA`, confirmation selects `Confirmed leaf mask (current ROI)`.
Vein length and specimen leaf area refer to the same region. Local ROIs give
local densities. External area remains a separate manual input (mm² or cm²);
select it explicitly and match it to the measured region and preparation state.
A pre-preparation area alone does not correct vein-length shrinkage. Without
scale calibration, mask area remains in px² and physical VLA is left blank.
In `6. Export`, click `Export Measurements` to measure the current labels and
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
and SHA-256, calibration, leaf area (confirmed mask pixel count, digest,
ROI scope, and mm²; or manual value and unit), the
parameters actually applied (ROI bounds, Gaussian sigma, Otsu and final
threshold, invert, every Clean Mask run), whether the mask was edited by hand
after the last automatic step, label pixel counts, number of traced paths, the
settings defaults in effect, and software versions. The vein mask itself is
saved as `*_vein_mask.npy` so hand edits are preserved. The confirmed leaf
region is saved as `*_leaf_mask.npy` with the label results. Whole-image path
exports require an explicit external area rather than an ROI mask denominator.

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
