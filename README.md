# papillary-ct-segmentation

Automated 3D segmentation of the left ventricular papillary muscles
(anterolateral, **APM**; posteromedial, **PPM**) and of the left ventricular
myocardium on contrast-enhanced cardiac CT, followed by PyRadiomics feature
extraction. Implemented as a batch script for [3D Slicer](https://www.slicer.org/).

This is the code used in:

> Olbryś P, et al. *Papillary Muscle Remodelling in Advanced HFrEF: Integrating
> Automated 3D Segmentation, CT Radiomics, and Plasma Proteomics.* (2026, in
> submission). <!-- TODO: add journal / DOI when available -->

The segmentation is rule based. Apart from the localization of the cardiac
chambers, which uses the open-source TotalSegmentator model, no model was
trained for this task; the papillary muscles are isolated by deterministic
morphological image processing. Any other method that yields a left
ventricular cavity mask that still contains the papillary muscles and
trabeculae can replace the localization step.

---

## Pipeline

| Step | Operation | Key parameters |
|---|---|---|
| 0 | Resampling of the CT volume to isotropic voxels (B-spline) | 0.75 mm |
| 1 | Chamber localization with TotalSegmentator (`heartchambers_highres`); intensity window applied inside the LV cavity mask to remove contrast-enhanced blood pool while keeping fatty and lean tissue | −190 … +242 HU (upper bound adjustable per case) |
| 2 | Multi-scale morphological opening of the thresholded LV tissue with a spherical structuring element; connected components scored by *volume × compactness × cavity depth*; the two best components that are angularly and spatially separated are the papillary cores. The scan stops at the largest radius that yields two cores (least aggressive erosion). | radius 3.5 → 1.0 mm, step 0.5 mm; min. core volume 0.10 mL; ≥ 45° about the LV centroid; ≥ 12 mm apart |
| 3 | Re-growth of the cores to the full thresholded tissue mask by a seeded watershed (discarded components enter as a third "debris" label); light clean-up (2-iteration erosion → largest component → 2-iteration dilation ∩ mask); anatomical labelling by the angle between each muscle's centroid vector and the LV→RV (septal) direction: the angularly farther muscle is the APM, the closer one the PPM. The LV myocardium minus both papillary masks is kept as a third region. | — |
| 4 | PyRadiomics feature extraction (SlicerRadiomics) for APM, PPM and myocardium: first-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM (shape is computed but was not analysed in the paper); re-segmentation to −190 … +242 HU; optional sweep over several fixed bin widths | `params/pm_radiomics_bin16.yaml`; bin widths 16 HU (main) or 2, 4, 8, 16, 32 HU (stability analysis) |

Fallbacks, both recorded per case in `processing_log.csv`:

* If no opening radius isolates two admissible cores, the muscles are split by
  a seeded watershed on the negative distance transform of the tissue mask
  (`split_method = distance_peaks_fallback`).
* If TotalSegmentator returns no right ventricle, APM/PPM are assigned by the
  anterior–posterior image axis (`labelling_method = image_axis_fallback`).
  This fallback is orientation dependent; check such cases visually.

If more than 70 % of the LV cavity survives the intensity window, residual
contrast-enhanced blood is assumed and the script asks whether to lower the
upper HU threshold, to correct the tissue mask manually in the Segment Editor,
to skip the case, or to abort the batch.

---

## Requirements

* 3D Slicer 5.x (the study used 5.12.3) with the extensions
  **TotalSegmentator** and **SlicerRadiomics** installed from the Extensions Manager.
* The TotalSegmentator task `heartchambers_highres` requires a free
  non-commercial licence key from the TotalSegmentator authors
  (see the TotalSegmentator module in Slicer → *Set license*). Without it the
  chamber localization step fails.
* Python packages inside Slicer: `numpy` and `scipy` are bundled;
  `scikit-image` is not — the script offers to install it on first start
  (`slicer.util.pip_install("scikit-image")`).
* A CUDA GPU is strongly recommended for TotalSegmentator; CPU execution works
  but takes several minutes per case.

## Usage

1. Export each CT study as a single `.nrrd` volume (Hounsfield units, contrast-enhanced, e.g. via Slicer's DICOM module) into one input folder.
2. In Slicer's Python console:

   ```python
   p = r"/path/to/papillary_segmentation.py"
   exec(open(p).read(), {"__file__": p, "__name__": "__main__"})
   ```

   or start Slicer with `Slicer --python-script /path/to/papillary_segmentation.py`.
3. In the window that opens, set
   * **Input folder** – the `.nrrd` volumes (files ending in `.seg.nrrd` are ignored; cases are processed in the numeric order of the first number in the file name),
   * **Output folder**,
   * **PyRadiomics parameter file** – defaults to `params/pm_radiomics_bin16.yaml`,
   * **Bin widths (HU)** – `16` reproduces the main analysis, `2,4,8,16,32` the bin-width stability analysis (each value overrides `binWidth` in the parameter file),
   * **Upper HU threshold** – default 242 HU,
   * optional saving of the TotalSegmentator chamber segmentation and of the resampled CT.
4. **Start batch**. **Stop after current case** interrupts the loop cleanly; the batch can be resumed later from the next file.

## Outputs (per case `<id>`)

| File | Content |
|---|---|
| `<id>_segmentation.seg.nrrd` | Segments `APM`, `PPM`, `Myocardium` on the 0.75 mm grid |
| `<id>_radiomics_bw<B>.tsv` | PyRadiomics features, one column per segment (`<id>_segment_APM`, …), one file per bin width *B* |
| `<id>_TotalSegmentator.seg.nrrd` | (optional) raw chamber segmentation |
| `<id>_CT_0.75mm.nrrd` | (optional) resampled CT used for all subsequent steps |
| `processing_log.csv` | One row per case: status, upper HU threshold used, manual correction flag, thresholded-tissue fraction of the LV, split method, opening radius, labelling method, APM/PPM angle to the septal vector, APM/PPM/myocardium volumes, bin widths, processing time |

The log is the provenance record for a batch; report the number of cases that
needed a threshold change, manual correction, or either fallback.

## Reproducing the analysis of the paper

* Default settings, bin widths `2,4,8,16,32`.
* The paper analysed first-order and texture features; shape features are in
  the TSV files but were excluded from every analysis (they are functions of
  the segmentation geometry, not of tissue).
* Radiomic values depend on the exact PyRadiomics version shipped with the
  SlicerRadiomics extension; the version used is printed in the Slicer Python
  console (`import radiomics; radiomics.__version__`).

## Structure of the code

`papillary_segmentation.py` has two parts:

* **Core algorithm** (top of the file, up to the `---- 3D Slicer` marker):
  pure NumPy / SciPy / scikit-image functions —
  `threshold_lv_tissue`, `find_papillary_cores`, `regrow_cores`,
  `split_by_distance_peaks`, `remove_trabecular_bridges`, `classify_apm_ppm`
  and the end-to-end `split_papillary_muscles`. Importable without Slicer.
* **Slicer application**: GUI, resampling, TotalSegmentator call, interactive
  failure handling, segmentation export, SlicerRadiomics job queue, logging.

All numeric parameters are module-level constants at the top of the file.

## Tests

```bash
pip install numpy scipy scikit-image pytest
python -m pytest tests/
```

The tests build a synthetic LV phantom (ellipsoidal cavity, two papillary
muscles, random trabeculae, a right ventricle) and check that both muscles are
recovered (Dice > 0.9 against the phantom), labelled consistently with the
septal direction, that the fallbacks run, and that residual blood pool is
detected.

## Limitations

* Designed for contrast-enhanced CT in which the blood pool is brighter than
  +242 HU; poorly enhanced studies need a lower upper threshold.
* Assumes two papillary muscle groups; accessory heads are merged into the
  nearest group.
* Chamber localization quality is that of TotalSegmentator; failures there
  propagate.

## Citation

See `CITATION.cff`. Please also cite 3D Slicer, TotalSegmentator and
PyRadiomics, on which this script depends.

## Licence

MIT — see `LICENSE`. TotalSegmentator's `heartchambers_highres` weights are
subject to their own (non-commercial) licence.
