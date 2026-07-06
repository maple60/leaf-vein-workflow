# Leaf Vein Workflow

napari plugin for leaf vein image workflow development.

## MVP

- Open an image with the standard napari image reader.
- Automatically create a rectangle ROI Shapes layer above the image.
- Automatically switch that Shapes layer to rectangle drawing mode.
- Add or update a cropped ROI Image layer only after clicking `Add ROI`.
- Create or update a vein mask Labels layer from the ROI crop with a manual
  threshold or an Otsu threshold estimate.

Saving cropped images, multiple ROIs, polygon ROIs, nD stacks, skeletonization,
and VLA calculation are outside the first MVP.

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
added. Draw one rectangle, then click `Add ROI` in the widget to create the ROI
crop layer. Click `Use Otsu` or enter a threshold manually, then click
`Create Mask` to create a vein mask layer from the crop.

Run tests:

```powershell
uv run pytest
```
