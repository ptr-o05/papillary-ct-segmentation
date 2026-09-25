"""
Automated 3D segmentation of the left ventricular papillary muscles (APM, PPM) and myocardium on contrast-enhanced cardiac CT, with PyRadiomics feature extraction — a 3D Slicer batch script.
Pipeline (see README.md and the Methods section of the accompanying paper):

  Step 0  Resample the CT volume to isotropic 0.75 mm spacing (B-spline).
  Step 1  Chamber localization with TotalSegmentator (task ``heartchambers_highres``), then intensity windowing of the left ventricular (LV) cavity mask to
          -190 ... +242 HU. This removes contrast-enhanced blood pool and keeps papillary/trabecular tissue (fatty and lean).
  Step 2  Multi-scale morphological opening of the thresholded LV tissue with a spherical structuring element (radius 3.5 -> 1.0 mm, step 0.5 mm).
          Connected components are scored by volume x compactness x cavity depth; the two best components that are >= 45 deg apart about the LV centroid and >= 12 mm apart are taken as the papillary cores.
  Step 3  Re-growth of the two cores to the full thresholded tissue mask with a seeded (flat-cost) watershed, a light morphological clean-up, and anatomical labelling: the muscle whose centroid vector is angularly
          farther from the LV->RV (septal) direction is the anterolateral papillary muscle (APM), the closer one the posteromedial (PPM). The LV myocardium (TotalSegmentator) with both papillary masks removed is kept as a third region.
  Step 4  PyRadiomics feature extraction (SlicerRadiomics) for APM, PPM and myocardium with the supplied parameter file, optionally repeated over several fixed bin widths.

NumPy / SciPy / scikit-image and can be imported and unit-tested outside 3D Slicer. The GUI and I/O section requires 3D Slicer with the TotalSegmentator and SlicerRadiomics extensions installed.

Usage inside 3D Slicer (Python console):

    exec(open(r"/path/to/papillary_segmentation.py").read())

or from the command line:

    Slicer --python-script /path/to/papillary_segmentation.py
"""

import csv
import glob
import os
import re
import shutil
import tempfile
import time

import numpy as np
from scipy import ndimage

try:
    from skimage import morphology as skmorph
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
except ImportError:
    skmorph = peak_local_max = watershed = None

# Segmentation parameters

RESAMPLE_SPACING_MM = 0.75          # isotropic voxel size after resampling
TOTALSEGMENTATOR_TASK = "heartchambers_highres"

HU_MIN = -190                       # lower bound of the tissue window
HU_MAX_DEFAULT = 242                # upper bound of the tissue window
BLOOD_POOL_FRACTION_LIMIT = 0.70    # tissue/LV voxel ratio above which residual blood pool is assumed (triggers a warning)
OPENING_RADIUS_MAX_MM = 3.5         # multi-scale opening: start radius
OPENING_RADIUS_MIN_MM = 1.0         # multi-scale opening: end radius
OPENING_RADIUS_STEP_MM = 0.5        # multi-scale opening: decrement
MIN_CORE_VOLUME_ML = 0.10           # smallest component considered a papillary muscle core
N_CANDIDATES_TO_SCORE = 10          # largest components scored per radius
MAX_DEPTH_BONUS_VOX = 10.0          # cap of the cavity-depth bonus (voxels)
CONSOLIDATION_ITERATIONS = 2        # dilation used to attach satellites to cores
MIN_CORE_ANGLE_DEG = 45.0           # minimum angular separation of the 2 cores
MIN_CORE_DISTANCE_MM = 12.0         # minimum Euclidean separation of the 2 cores
CLEANUP_ITERATIONS = 2              # erosion/dilation of the final clean-up

# Fallback (distance-transform peaks), used only if no opening radius isolates two cores

FALLBACK_PEAK_MIN_DISTANCE_VOX = 3
FALLBACK_SAFE_MARGIN_MM = 10.0

SEGMENT_NAMES = {"apm": "APM", "ppm": "PPM", "myo": "Myocardium"}
SEGMENT_COLORS = {"apm": (0.75, 0.22, 0.17), "ppm": (0.14, 0.24, 0.55),
                  "myo": (0.85, 0.65, 0.55)}


class SplitError(ValueError):
    # Raised when APM/PPM separation fails for a recoverable reason
    pass

# Core algorithm

def threshold_lv_tissue(image_hu, lv_mask, hu_min=HU_MIN, hu_max=HU_MAX_DEFAULT):
    # Voxels of the LV cavity mask whose intensity lies in min-max HU range
    # Contrast-enhanced blood pool lies above max HU threshold; fatty (from -190 to -30 HU) and lean (from 0 to 242 HU) tissue is retained, bounds are inclusive

    return (lv_mask > 0) & (image_hu >= hu_min) & (image_hu <= hu_max)


def compactness_score(component_mask):
    # 1/elongation of a component, from the eigenvalues of the covariance matrix of its voxel coordinates; 1 for an isotropic blob; 0 for a thin bridge (trabecula)
    coords = np.argwhere(component_mask)
    if len(coords) < 4:
        return 0.0
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 1e-6, None)
    return float(eigvals.min() / eigvals.max())


def _angle_deg(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def find_papillary_cores(tissue_mask, voxel_mm, lv_depth_map, lv_center,
                         min_radius_mm=OPENING_RADIUS_MIN_MM,
                         max_radius_mm=OPENING_RADIUS_MAX_MM,
                         step_mm=OPENING_RADIUS_STEP_MM,
                         min_core_volume_ml=MIN_CORE_VOLUME_ML,
                         n_candidates_to_score=N_CANDIDATES_TO_SCORE,
                         max_depth_bonus_vox=MAX_DEPTH_BONUS_VOX,
                         consolidation_iterations=CONSOLIDATION_ITERATIONS,
                         min_angle_deg=MIN_CORE_ANGLE_DEG,
                         min_distance_mm=MIN_CORE_DISTANCE_MM):

    # Step 2: multi-scale morphological isolation of the two papillary cores
    # Returns '(core1, core2, discarded_components, radius_mm)' for the largest opening radius at which two admissible cores exist, or 'None' if no radius satisfies the criterion

    min_core_vox = (min_core_volume_ml * 1000.0) / (voxel_mm ** 3)
    r_mm = max_radius_mm

    while r_mm >= min_radius_mm - 1e-9:
        r_vox = max(1, int(round(r_mm / voxel_mm)))
        opened = ndimage.binary_opening(tissue_mask, structure=skmorph.ball(r_vox))
        labeled, n = ndimage.label(opened)

        if n >= 2:
            sizes = ndimage.sum(opened, labeled, range(1, n + 1))
            top_idx = np.argsort(sizes)[::-1][:n_candidates_to_score]

            scored = []
            for idx in top_idx:
                vol_vox = sizes[idx]
                if vol_vox < min_core_vox:
                    continue
                comp = labeled == (idx + 1)
                centroid = tuple(np.round(ndimage.center_of_mass(comp)).astype(int))
                depth = min(lv_depth_map[centroid], max_depth_bonus_vox)
                score = vol_vox * compactness_score(comp) * (1.0 + depth)
                scored.append({"score": score, "mask": comp, "centroid": np.array(centroid)})

            if len(scored) >= 2:
                scored.sort(key=lambda s: s["score"], reverse=True)
                core1 = scored[0]
                vec1 = core1["centroid"] - np.asarray(lv_center)

                core2_idx = -1
                for i in range(1, len(scored)):
                    cand = scored[i]
                    vec2 = cand["centroid"] - np.asarray(lv_center)
                    dist_mm = np.linalg.norm(cand["centroid"] - core1["centroid"]) * voxel_mm
                    if _angle_deg(vec1, vec2) >= min_angle_deg and dist_mm >= min_distance_mm:
                        core2_idx = i
                        break

                if core2_idx > 0:
                    core1_mask = core1["mask"].copy()
                    core2_mask = scored[core2_idx]["mask"].copy()
                    zone1 = ndimage.binary_dilation(core1_mask, iterations=consolidation_iterations) & tissue_mask
                    zone2 = ndimage.binary_dilation(core2_mask, iterations=consolidation_iterations) & tissue_mask

                    discarded = []
                    for i, cand in enumerate(scored):
                        if i in (0, core2_idx):
                            continue
                        if np.any(cand["mask"] & zone1):
                            core1_mask |= cand["mask"]
                        elif np.any(cand["mask"] & zone2):
                            core2_mask |= cand["mask"]
                        else:
                            discarded.append(cand["mask"])
                    return core1_mask, core2_mask, discarded, r_mm

        r_mm -= step_mm

    return None

# Step 3a: re-grow the eroded cores to the full thresholded tissue mask
# A watershed on a flat cost image seeded with the cores is a geodesic nearest-seed partition of the tissue mask
# Discarded components are entered as a third ('debris') label so that they do not get attached to either muscle; returns an integer label map (1 = core1, 2 = core2, 3 = debris)

def regrow_cores(tissue_mask, core1, core2, discarded):
    markers = np.zeros(tissue_mask.shape, dtype=np.int32)
    for d in discarded:
        markers[d] = 3
    markers[core1] = 1
    markers[core2] = 2
    flat_cost = np.zeros(tissue_mask.shape, dtype=np.float32)
    return watershed(flat_cost, markers=markers, mask=tissue_mask)

# Fallback for Step 2/3: seeded watershed on the negative distance transform of the tissue mask, with seeds at the two thickest, deepest, sufficiently separated local maxima
# Used only when find_papillary_cores() returns 'None'

def split_by_distance_peaks(tissue_mask, lv_depth_map, voxel_mm, lv_center,
                            min_angle_deg=MIN_CORE_ANGLE_DEG,
                            min_distance_mm=MIN_CORE_DISTANCE_MM):
    dist_map = ndimage.distance_transform_edt(tissue_mask)
    smooth = ndimage.gaussian_filter(dist_map, sigma=1.0)
    peaks = peak_local_max(smooth, min_distance=FALLBACK_PEAK_MIN_DISTANCE_VOX)
    if len(peaks) < 2:
        raise SplitError("Fewer than two tissue islands detected")

    scored = sorted(
        ({"coord": np.array(p), "score": dist_map[tuple(p)] * lv_depth_map[tuple(p)]} for p in peaks),
        key=lambda s: s["score"], reverse=True)

    m1 = scored[0]["coord"]
    vec1 = m1 - np.asarray(lv_center)
    m2_idx = -1
    for i in range(1, len(scored)):
        cand = scored[i]["coord"]
        vec2 = cand - np.asarray(lv_center)
        dist_mm = np.linalg.norm(cand - m1) * voxel_mm
        if _angle_deg(vec1, vec2) >= min_angle_deg and dist_mm >= min_distance_mm:
            m2_idx = i
            break
    if m2_idx < 0:
        raise SplitError("Second papillary muscle not found (separation failed)")
    m2 = scored[m2_idx]["coord"]

    markers = np.zeros(tissue_mask.shape, dtype=np.int32)
    markers[tuple(m1)] = 1
    markers[tuple(m2)] = 2
    safe_margin_vox = int(FALLBACK_SAFE_MARGIN_MM / voxel_mm)
    for i, s in enumerate(scored):
        if i in (0, m2_idx):
            continue
        p = s["coord"]
        if np.linalg.norm(p - m1) > safe_margin_vox and np.linalg.norm(p - m2) > safe_margin_vox:
            markers[tuple(p)] = 3
    return watershed(-dist_map, markers=markers, mask=tissue_mask)

# Step 3b: light clean-up: erode, keep the largest connected component, dilate back and intersect with the original mask
# Removes thin trabecular bridges left attached after the watershed; if erosion removes everything the mask is returned unchanged

def remove_trabecular_bridges(mask, iterations=CLEANUP_ITERATIONS):
    eroded = ndimage.binary_erosion(mask, iterations=iterations)
    labeled, n = ndimage.label(eroded)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    main = labeled == sizes.argmax()
    return ndimage.binary_dilation(main, iterations=iterations) & mask

# Step 3c: anatomical labelling by angle to the septal (LV -> RV) vector
# Returns '(label_of_1, label_of_2, angle_1, angle_2)' or 'None' if no septal vector is available
# The muscle angularly farther from the septal direction is the APM (anterolateral), the closer one the PPM (posteromedial)

def classify_apm_ppm(center1, center2, lv_center, septal_vector):
    if septal_vector is None or np.linalg.norm(septal_vector) < 1e-6:
        return None
    a1 = _angle_deg(np.asarray(center1) - np.asarray(lv_center), septal_vector)
    a2 = _angle_deg(np.asarray(center2) - np.asarray(lv_center), septal_vector)
    if a1 < a2:
        return "PPM", "APM", a1, a2
    return "APM", "PPM", a1, a2

# Steps 2-3 end to end:
# :param tissue_mask: boolean array, thresholded LV tissue
# :param lv_mask: boolean array, full LV cavity mask (blood pool + tissue)
# :param voxel_mm: isotropic voxel size in mm
# :param septal_vector: LV-centroid -> RV-centroid vector in voxel index space, or 'None'
# :param check_blood_pool: raise :class:'SplitError' when the tissue mask occupies more than 'BLOOD_POOL_FRACTION_LIMIT' of the LV (residual contrast-enhanced blood); disabled after manual correction
# :returns: dict with boolean masks 'apm'/'ppm' (cleaned) and QC fields

def split_papillary_muscles(tissue_mask, lv_mask, voxel_mm, septal_vector=None,
                            check_blood_pool=True):
    tissue_mask = tissue_mask.astype(bool)
    lv_mask = lv_mask.astype(bool)

    n_tissue, n_lv = int(tissue_mask.sum()), int(lv_mask.sum())
    tissue_fraction = n_tissue / n_lv if n_lv else float("nan")
    if check_blood_pool and n_lv and tissue_fraction > BLOOD_POOL_FRACTION_LIMIT:
        raise SplitError(
            f"Residual blood pool: thresholded tissue is {tissue_fraction * 100:.0f}% of the LV "
            f"cavity (> {BLOOD_POOL_FRACTION_LIMIT * 100:.0f}%), lower the upper HU threshold")
    if n_tissue == 0:
        raise SplitError("Thresholded LV tissue mask is empty")

    lv_center = np.array(ndimage.center_of_mass(lv_mask))
    lv_depth_map = ndimage.distance_transform_edt(lv_mask)

    cores = find_papillary_cores(tissue_mask, voxel_mm, lv_depth_map, lv_center)
    if cores is not None:
        core1, core2, discarded, radius_mm = cores
        labels = regrow_cores(tissue_mask, core1, core2, discarded)
        split_method = "multiscale_opening"
    else:
        labels = split_by_distance_peaks(tissue_mask, lv_depth_map, voxel_mm, lv_center)
        radius_mm = float("nan")
        split_method = "distance_peaks_fallback"

    mask1, mask2 = labels == 1, labels == 2
    if not mask1.any() or not mask2.any():
        raise SplitError("Watershed produced an empty muscle label")
    c1, c2 = ndimage.center_of_mass(mask1), ndimage.center_of_mass(mask2)

    classification = classify_apm_ppm(c1, c2, lv_center, septal_vector)
    if classification is not None:
        name1, name2, angle1, angle2 = classification
        labelling_method = "septal_angle"
    else:
        # Orientation-dependent fallback (no RV available): in the default
        # Slicer array layout (K, J, I) axis 1 runs anterior -> posterior, so the more anterior muscle is called APM
        name1, name2 = ("APM", "PPM") if c1[1] < c2[1] else ("PPM", "APM")
        angle1 = angle2 = float("nan")
        labelling_method = "image_axis_fallback"

    if name1 == "APM":
        apm, ppm, apm_angle, ppm_angle = mask1, mask2, angle1, angle2
    else:
        apm, ppm, apm_angle, ppm_angle = mask2, mask1, angle2, angle1

    return {
        "apm": remove_trabecular_bridges(apm),
        "ppm": remove_trabecular_bridges(ppm),
        "split_method": split_method,
        "opening_radius_mm": radius_mm,
        "labelling_method": labelling_method,
        "apm_angle_to_septum_deg": apm_angle,
        "ppm_angle_to_septum_deg": ppm_angle,
        "tissue_fraction_of_lv": tissue_fraction,
    }

# 3D Slicer batch application (GUI, I/O, TotalSegmentator, radiomics)

try:
    import slicer
    import qt
    import ctk
except ImportError:
    slicer = qt = ctk = None


LOG_COLUMNS = [
    "case_id", "status", "upper_threshold_HU", "manual_correction",
    "tissue_fraction_of_lv", "split_method", "opening_radius_mm",
    "labelling_method", "apm_angle_to_septum_deg", "ppm_angle_to_septum_deg",
    "apm_volume_ml", "ppm_volume_ml", "myocardium_volume_ml",
    "bin_widths_HU", "processing_time_s", "timestamp",
]


def _ensure_scikit_image():
    # scikit-image is not bundled with 3D Slicer; offer to install it
    global skmorph, peak_local_max, watershed
    if skmorph is not None:
        return True
    if not slicer.util.confirmOkCancelDisplay(
            "This script requires the Python package 'scikit-image', which is not "
            "installed in this 3D Slicer. Install it now?"):
        return False
    slicer.util.pip_install("scikit-image")
    from skimage import morphology as skmorph
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    return True


def _find_segment_id(segmentation, *name_fragments):
    # First segment whose lower-case name contains all given fragments
    for i in range(segmentation.GetNumberOfSegments()):
        s_id = segmentation.GetNthSegmentID(i)
        name = segmentation.GetSegment(s_id).GetName().lower()
        if all(f in name for f in name_fragments):
            return s_id
    return ""


def _parameter_file_for_bin_width(base_param_path, bin_width, temp_dir):
    # Copy of the PyRadiomics parameter file with 'binWidth' replaced.
    # Uses a text substitution so that no YAML library is required. The file must already contain a 'binWidth:' entry.
    with open(base_param_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if not re.search(r"(?m)^\s*binWidth\s*:", text):
        raise ValueError("Parameter file has no 'binWidth' entry; cannot run a bin-width sweep")
    new_text = re.sub(r"(?m)^(\s*binWidth\s*:\s*)\S+", lambda m: f"{m.group(1)}{bin_width}", text)
    out_path = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(base_param_path))[0]}_bw{bin_width}.yaml")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    return out_path


class PapillaryMuscleSegmentationWidget(qt.QWidget if qt else object):
    # Batch GUI: iterates over all *.nrrd CT volumes in an input folder

    def __init__(self):
        super().__init__()
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        self.default_param_file = os.path.join(script_dir, "params", "pm_radiomics_bin16.yaml")

        self.files = []
        self.current_index = 0
        self.is_running = False
        self.threshold_overrides = {}

        # per-case state
        self.case_id = None
        self.case_t0 = None
        self.volume = None
        self.ts_node = None
        self.seg_node = None
        self.lv_arr = self.rv_arr = self.myo_arr = None
        self.split_result = None
        self.manual_correction = False
        self.radiomics_jobs = []
        self.radiomics_logic = None
        self.radiomics_table = None
        self.temp_dir = tempfile.mkdtemp(prefix="pm_seg_")

        self._build_ui()

    # UI
    def _build_ui(self):
        layout = qt.QFormLayout()

        self.input_dir_edit = ctk.ctkPathLineEdit()
        self.input_dir_edit.filters = ctk.ctkPathLineEdit.Dirs
        self.input_dir_edit.setToolTip("Folder with contrast-enhanced cardiac CT volumes (*.nrrd). "
                                       "Files ending in .seg.nrrd are ignored.")
        self.input_dir_edit.connect("currentPathChanged(QString)", self._refresh_file_list)
        layout.addRow("Input folder (CT *.nrrd):", self.input_dir_edit)

        self.output_dir_edit = ctk.ctkPathLineEdit()
        self.output_dir_edit.filters = ctk.ctkPathLineEdit.Dirs
        self.output_dir_edit.currentPath = os.path.join(os.path.expanduser("~"), "papillary_segmentation_output")
        layout.addRow("Output folder:", self.output_dir_edit)

        self.param_file_edit = ctk.ctkPathLineEdit()
        self.param_file_edit.filters = ctk.ctkPathLineEdit.Files
        self.param_file_edit.nameFilters = ["PyRadiomics parameter file (*.yaml *.yml)"]
        self.param_file_edit.currentPath = self.default_param_file
        layout.addRow("PyRadiomics parameter file:", self.param_file_edit)

        self.bin_widths_edit = qt.QLineEdit("16")
        self.bin_widths_edit.setToolTip("Comma-separated fixed bin widths in HU. The publication used 16 HU for the "
                                        "main analysis and 2, 4, 8, 16, 32 HU for the stability analysis. "
                                        "Each value overrides 'binWidth' in the parameter file.")
        layout.addRow("Bin widths (HU):", self.bin_widths_edit)

        self.hu_max_spin = qt.QSpinBox()
        self.hu_max_spin.setRange(HU_MIN, 1000)
        self.hu_max_spin.setValue(HU_MAX_DEFAULT)
        self.hu_max_spin.setToolTip("Upper bound of the tissue window inside the LV cavity. Lower it when "
                                    "contrast-enhanced blood remains in the segmented tissue.")
        layout.addRow(f"Upper HU threshold (lower = {HU_MIN} HU):", self.hu_max_spin)

        self.save_ts_check = qt.QCheckBox("Save TotalSegmentator chamber segmentation")
        self.save_ts_check.setChecked(True)
        layout.addRow(self.save_ts_check)

        self.save_resampled_check = qt.QCheckBox(f"Save resampled CT volume ({RESAMPLE_SPACING_MM} mm)")
        self.save_resampled_check.setChecked(False)
        layout.addRow(self.save_resampled_check)

        self.run_radiomics_check = qt.QCheckBox("Extract radiomic features (SlicerRadiomics)")
        self.run_radiomics_check.setChecked(True)
        layout.addRow(self.run_radiomics_check)

        self.status_label = qt.QLabel("Select an input folder.")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.status_label.setWordWrap(True)
        layout.addRow(self.status_label)

        self.btn_start = qt.QPushButton("Start batch")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-weight: bold;")
        self.btn_start.clicked.connect(self.start_batch)
        layout.addRow(self.btn_start)

        self.btn_stop = qt.QPushButton("Stop after current case")
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; padding: 6px; font-weight: bold;")
        self.btn_stop.clicked.connect(lambda: self.stop_batch())
        self.btn_stop.setEnabled(False)
        layout.addRow(self.btn_stop)

        self.setLayout(layout)
        self.setWindowTitle("Papillary muscle segmentation and radiomics (APM / PPM / myocardium)")
        self.resize(640, 380)
        self.setWindowFlags(qt.Qt.WindowStaysOnTopHint)
        self.show()

    def update_status(self, text):
        self.status_label.setText(text)
        slicer.app.processEvents()

    def _refresh_file_list(self, *_):
        input_dir = self.input_dir_edit.currentPath
        files = [f for f in glob.glob(os.path.join(input_dir, "*.nrrd")) if not f.endswith(".seg.nrrd")]

        def sort_key(path):
            numbers = re.findall(r"\d+", os.path.basename(path))
            return (int(numbers[0]) if numbers else 0, os.path.basename(path))

        self.files = sorted(files, key=sort_key)
        self.current_index = 0
        self.update_status(f"Found {len(self.files)} CT volume(s). Ready.")

    # batch control
    def _parse_bin_widths(self):
        values = [v.strip() for v in self.bin_widths_edit.text.split(",") if v.strip()]
        bin_widths = sorted({int(v) for v in values})
        if not bin_widths or any(b <= 0 for b in bin_widths):
            raise ValueError("Bin widths must be positive integers, e.g. '16' or '2,4,8,16,32'")
        return bin_widths

    def start_batch(self):
        if not self.files:
            self._refresh_file_list()
        if self.current_index >= len(self.files):
            self.update_status("No CT volumes left to process.")
            return
        if not _ensure_scikit_image():
            return
        for module_name in ("TotalSegmentator", "SlicerRadiomics"):
            if module_name == "SlicerRadiomics" and not self.run_radiomics_check.checked:
                continue
            if not hasattr(slicer.modules, module_name.lower()):
                self.update_status(f"ERROR: the {module_name} extension is not installed.")
                return
        if self.run_radiomics_check.checked:
            if not os.path.isfile(self.param_file_edit.currentPath):
                self.update_status("ERROR: PyRadiomics parameter file not found.")
                return
            try:
                self._parse_bin_widths()
            except ValueError as e:
                self.update_status(f"ERROR: {e}")
                return
        os.makedirs(self.output_dir_edit.currentPath, exist_ok=True)

        self.is_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.process_case()

    def stop_batch(self, message=None):
        self.is_running = False
        self.btn_stop.setEnabled(False)
        self.update_status(message or "Finishing the current case, then stopping.")

    def _next_case(self):
        self.current_index += 1
        if self.is_running:
            qt.QTimer.singleShot(500, self.process_case)
        else:
            self.update_status(f"Stopped. Next case would be file #{self.current_index + 1}.")
            self.btn_start.setEnabled(True)

    def _fail_case(self, message):
        """Log a failed case and continue with the next one."""
        self._write_log_row(status=f"failed: {message}")
        self.update_status(f"{self.case_id}: FAILED - {message}")
        print(f"[papillary_segmentation] {self.case_id}: FAILED - {message}")
        self._next_case()

    # per case
    def process_case(self):
        if not self.is_running:
            self.btn_start.setEnabled(True)
            self.update_status(f"Stopped. Next case would be file #{self.current_index + 1}.")
            return
        if self.current_index >= len(self.files):
            self.is_running = False
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.update_status("All CT volumes processed.")
            return

        path = self.files[self.current_index]
        self.case_id = os.path.basename(path)[:-len(".nrrd")]
        self.case_t0 = time.time()
        self.split_result = None
        self.manual_correction = False

        slicer.mrmlScene.Clear(0)
        native = slicer.util.loadVolume(path)
        if not native:
            self._fail_case("could not load volume")
            return

        # Step 0: isotropic resampling
        self.update_status(f"{self.case_id}\nStep 0/4: resampling to {RESAMPLE_SPACING_MM} mm isotropic")
        try:
            self.volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{self.case_id}_resampled")
            params = {"InputVolume": native.GetID(), "OutputVolume": self.volume.GetID(),
                      "spacing": [RESAMPLE_SPACING_MM] * 3, "interpolationType": "bspline"}
            slicer.cli.runSync(slicer.modules.resamplescalarvolume, None, params)
            slicer.mrmlScene.RemoveNode(native)
        except Exception as e:
            self._fail_case(f"resampling: {e}")
            return

        # Step 1a: chamber localization
        self.update_status(f"{self.case_id}\nStep 1/4: TotalSegmentator ({TOTALSEGMENTATOR_TASK})")
        try:
            import TotalSegmentator
            self.ts_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{self.case_id}_chambers")
            # Only 'task' is passed so that the call works with both the older (fast=) and the current (quality=) signature of the extension
            TotalSegmentator.TotalSegmentatorLogic().process(self.volume, self.ts_node, task=TOTALSEGMENTATOR_TASK)
        except Exception as e:
            self._fail_case(f"TotalSegmentator: {e}")
            return

        segmentation = self.ts_node.GetSegmentation()
        lv_id = _find_segment_id(segmentation, "left ventricle") or _find_segment_id(segmentation, "ventricle_left")
        rv_id = _find_segment_id(segmentation, "right ventricle") or _find_segment_id(segmentation, "ventricle_right")
        myo_id = _find_segment_id(segmentation, "myocardium")
        if not lv_id:
            self._fail_case("no left ventricle segment in TotalSegmentator output")
            return
        if not myo_id:
            self._fail_case("no myocardium segment in TotalSegmentator output")
            return

        self.lv_arr = slicer.util.arrayFromSegmentBinaryLabelmap(self.ts_node, lv_id, self.volume) > 0
        self.myo_arr = slicer.util.arrayFromSegmentBinaryLabelmap(self.ts_node, myo_id, self.volume) > 0
        self.rv_arr = (slicer.util.arrayFromSegmentBinaryLabelmap(self.ts_node, rv_id, self.volume) > 0) if rv_id else None
        if self.rv_arr is not None and not self.rv_arr.any():
            self.rv_arr = None

        if self.save_ts_check.checked:
            slicer.util.saveNode(self.ts_node, os.path.join(self.output_dir_edit.currentPath,
                                                            f"{self.case_id}_TotalSegmentator.seg.nrrd"))
        if self.save_resampled_check.checked:
            slicer.util.saveNode(self.volume, os.path.join(self.output_dir_edit.currentPath,
                                                           f"{self.case_id}_CT_{RESAMPLE_SPACING_MM}mm.nrrd"))

        self.run_split_step()

    def run_split_step(self):
        """Steps 1b-3: thresholding, APM/PPM separation, interactive recovery."""
        spacing = self.volume.GetSpacing()
        if max(spacing) - min(spacing) > 1e-3:
            self._fail_case(f"volume is not isotropic after resampling: {spacing}")
            return
        voxel_mm = spacing[0]
        image = slicer.util.arrayFromVolume(self.volume)

        septal_vector = None
        if self.rv_arr is not None:
            septal_vector = (np.array(ndimage.center_of_mass(self.rv_arr))
                             - np.array(ndimage.center_of_mass(self.lv_arr)))

        hu_max = self.threshold_overrides.get(self.case_id, self.hu_max_spin.value)
        while True:
            self.update_status(f"{self.case_id}\nStep 1/4: thresholding LV cavity to [{HU_MIN}, {hu_max}] HU")
            tissue = threshold_lv_tissue(image, self.lv_arr, HU_MIN, hu_max)

            self.update_status(f"{self.case_id}\nStep 2-3/4: papillary muscle separation")
            try:
                self.split_result = split_papillary_muscles(tissue, self.lv_arr, voxel_mm, septal_vector)
                break
            except SplitError as e:
                action = self._ask_on_failure(str(e), hu_max)
                if action == "retry":
                    hu_max = self.threshold_overrides[self.case_id]
                    continue
                if action == "manual":
                    tissue = self._manual_correction(tissue)
                    try:
                        self.split_result = split_papillary_muscles(tissue, self.lv_arr, voxel_mm, septal_vector,
                                                                    check_blood_pool=False)
                        self.manual_correction = True
                        break
                    except Exception as e2:
                        self._fail_case(f"separation after manual correction: {e2}")
                        return
                if action == "skip":
                    self._fail_case("skipped by user")
                    return
                self.stop_batch(f"Batch aborted at {self.case_id}.")
                self.btn_start.setEnabled(True)
                return
            except Exception as e:
                self._fail_case(f"separation: {e}")
                return

        self.split_result["upper_threshold_HU"] = hu_max
        self.build_segmentation_node()

    def _ask_on_failure(self, error_msg, hu_max):
        msg = qt.QMessageBox()
        msg.setIcon(qt.QMessageBox.Warning)
        msg.setWindowTitle(f"APM/PPM separation failed: {self.case_id}")
        msg.setText(f"{error_msg}\n\nCurrent upper HU threshold: {hu_max} HU\n\nWhat would you like to do?")
        btn_retry = msg.addButton("Retry with another HU threshold", qt.QMessageBox.ActionRole)
        btn_manual = msg.addButton("Correct manually in Segment Editor", qt.QMessageBox.ActionRole)
        btn_skip = msg.addButton("Skip this case", qt.QMessageBox.ActionRole)
        msg.addButton("Abort batch", qt.QMessageBox.RejectRole)
        msg.exec_()
        clicked = msg.clickedButton()
        if clicked == btn_retry:
            dlg = qt.QInputDialog()
            dlg.setWindowTitle("New upper HU threshold")
            dlg.setLabelText("Upper bound of the tissue window (a lower value removes more\n"
                             "weakly enhanced blood from the LV cavity):")
            dlg.setInputMode(qt.QInputDialog.IntInput)
            dlg.setIntRange(HU_MIN, 1000)
            dlg.setIntStep(1)
            dlg.setIntValue(max(HU_MIN, hu_max - 20))
            if dlg.exec_() == qt.QDialog.Accepted:
                self.threshold_overrides[self.case_id] = dlg.intValue()
                return "retry"
            return "abort"
        if clicked == btn_manual:
            return "manual"
        if clicked == btn_skip:
            return "skip"
        return "abort"

    def _manual_correction(self, tissue):
        """Let the user edit the thresholded tissue mask in Segment Editor."""
        work_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{self.case_id}_manual")
        work_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume)
        seg_id = work_node.GetSegmentation().AddEmptySegment("", "LV_tissue")
        slicer.util.updateSegmentBinaryLabelmapFromArray(tissue.astype(np.uint8), work_node, seg_id, self.volume)

        slicer.util.selectModule("SegmentEditor")
        editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
        editor.setMRMLScene(slicer.mrmlScene)
        editor.setSegmentationNode(work_node)
        editor.setSourceVolumeNode(self.volume)
        editor.setCurrentSegmentID(seg_id)

        msg = qt.QMessageBox()
        msg.setWindowTitle("Manual correction")
        msg.setText(f"Edit the segment 'LV_tissue' of case {self.case_id} with the Threshold / Paint / "
                    "Scissors / Islands tools in the Segment Editor (main window).\n\n"
                    "Click OK to resume the automatic APM/PPM separation on the corrected mask.")
        msg.setStandardButtons(qt.QMessageBox.Ok)
        msg.exec_()

        corrected = slicer.util.arrayFromSegmentBinaryLabelmap(work_node, seg_id, self.volume) > 0
        slicer.mrmlScene.RemoveNode(work_node)
        return corrected & self.lv_arr

    def build_segmentation_node(self):
        """Assemble APM, PPM and myocardium (minus papillary muscles) segments."""
        apm, ppm = self.split_result["apm"], self.split_result["ppm"]
        myo = self.myo_arr & ~apm & ~ppm   # TotalSegmentator labels are disjoint; explicit for safety

        voxel_ml = float(np.prod(self.volume.GetSpacing())) / 1000.0
        self.split_result["apm_volume_ml"] = apm.sum() * voxel_ml
        self.split_result["ppm_volume_ml"] = ppm.sum() * voxel_ml
        self.split_result["myocardium_volume_ml"] = myo.sum() * voxel_ml

        self.seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", self.case_id)
        self.seg_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume)
        segmentation = self.seg_node.GetSegmentation()
        for key, mask in (("apm", apm), ("ppm", ppm), ("myo", myo)):
            seg_id = segmentation.AddEmptySegment("", SEGMENT_NAMES[key])
            segmentation.GetSegment(seg_id).SetColor(*SEGMENT_COLORS[key])
            slicer.util.updateSegmentBinaryLabelmapFromArray(mask.astype(np.uint8), self.seg_node, seg_id, self.volume)

        slicer.util.saveNode(self.seg_node, os.path.join(self.output_dir_edit.currentPath,
                                                         f"{self.case_id}_segmentation.seg.nrrd"))
        slicer.mrmlScene.RemoveNode(self.ts_node)
        self.ts_node = None

        if self.run_radiomics_check.checked:
            self.start_radiomics()
        else:
            self.finish_case()

    # radiomics
    def start_radiomics(self):
        base = self.param_file_edit.currentPath
        self.radiomics_jobs = []
        try:
            for bw in self._parse_bin_widths():
                param_path = _parameter_file_for_bin_width(base, bw, self.temp_dir)
                out_path = os.path.join(self.output_dir_edit.currentPath, f"{self.case_id}_radiomics_bw{bw}.tsv")
                self.radiomics_jobs.append((bw, param_path, out_path))
        except Exception as e:
            self._fail_case(f"radiomics parameter file: {e}")
            return
        self._run_next_radiomics_job()

    def _run_next_radiomics_job(self):
        if not self.radiomics_jobs:
            self.finish_case()
            return
        bw, param_path, out_path = self.radiomics_jobs[0]
        self.update_status(f"{self.case_id}\nStep 4/4: radiomics (bin width {bw} HU), "
                           f"{len(self.radiomics_jobs)} job(s) left")
        try:
            import SlicerRadiomics
            self.radiomics_logic = SlicerRadiomics.SlicerRadiomicsLogic()
            self.radiomics_table = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode",
                                                                      f"{self.case_id}_radiomics_bw{bw}")
            self.radiomics_logic.runCLIWithParameterFile(self.volume, self.seg_node, self.radiomics_table,
                                                         param_path, self._on_radiomics_job_finished)
        except Exception as e:
            self._fail_case(f"radiomics: {e}")

    def _on_radiomics_job_finished(self):
        bw, param_path, out_path = self.radiomics_jobs.pop(0)
        slicer.util.saveNode(self.radiomics_table, out_path)
        slicer.mrmlScene.RemoveNode(self.radiomics_table)
        self.radiomics_table = None
        self._run_next_radiomics_job()

    # finalize
    def _write_log_row(self, status="ok"):
        log_path = os.path.join(self.output_dir_edit.currentPath, "processing_log.csv")
        r = self.split_result or {}
        row = {
            "case_id": self.case_id,
            "status": status,
            "upper_threshold_HU": r.get("upper_threshold_HU", self.threshold_overrides.get(self.case_id, self.hu_max_spin.value)),
            "manual_correction": self.manual_correction,
            "tissue_fraction_of_lv": r.get("tissue_fraction_of_lv", ""),
            "split_method": r.get("split_method", ""),
            "opening_radius_mm": r.get("opening_radius_mm", ""),
            "labelling_method": r.get("labelling_method", ""),
            "apm_angle_to_septum_deg": r.get("apm_angle_to_septum_deg", ""),
            "ppm_angle_to_septum_deg": r.get("ppm_angle_to_septum_deg", ""),
            "apm_volume_ml": r.get("apm_volume_ml", ""),
            "ppm_volume_ml": r.get("ppm_volume_ml", ""),
            "myocardium_volume_ml": r.get("myocardium_volume_ml", ""),
            "bin_widths_HU": self.bin_widths_edit.text if self.run_radiomics_check.checked else "",
            "processing_time_s": round(time.time() - self.case_t0, 1) if self.case_t0 else "",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_header = not os.path.isfile(log_path)
        with open(log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def finish_case(self):
        self._write_log_row(status="ok")
        if self.seg_node is not None:
            slicer.mrmlScene.RemoveNode(self.seg_node)
            self.seg_node = None
        self.update_status(f"{self.case_id}: done ({self.split_result['split_method']}, "
                           f"{self.split_result['labelling_method']}).")
        self._next_case()

    def closeEvent(self, event):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        event.accept()


if slicer is not None:
    slicer.modules.papillary_segmentation_widget = PapillaryMuscleSegmentationWidget()
