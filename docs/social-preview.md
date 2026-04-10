# GitHub repository social preview

GitHub does **not** read a special file from the repo for the repository link card. A maintainer must upload an image under **Settings → General → Social preview**. Until that is done, shared links may use a generic or auto-generated preview.

## What to upload

Use the committed PNG:

- **File:** `docs/assets/open-graph-cover.png`  
- **Size:** 1280×640 (matches what `scripts/export_open_graph_cover.py` exports)

If that PNG is missing or you need to regenerate it from the scripted layout, from the repository root run:

```bash
python scripts/export_open_graph_cover.py
```

That writes the default path `docs/assets/open-graph-cover.png` (requires [Pillow](https://pillow.readthedocs.io/); install with the project as usual).

The editable vector reference for the same design lives at `docs/assets/open-graph-cover.svg`; the export script renders the cover programmatically for consistent PNG output across environments.

## Where to set it in GitHub

1. Open the repository on GitHub.
2. Go to **Settings** (repository settings, not your account settings).
3. In the left sidebar, select **General**.
4. Scroll to **Social preview**.
5. Choose **Edit** (or **Upload an image**), select `open-graph-cover.png`, save if prompted.

Caching on social platforms can delay updates after you change the image.
