# Hyperspectral Pointcloud Fusion

`hyperspectral_pointcloud_fusion` converts multi-view directional hyperspectral imagery and a registered point cloud into two complementary 3-D spectral products:

1. a weighted fused spectrum for each point; and
2. an all-view directional BRF observation set that preserves every retained point-view observation.

The workflow includes scene inventory, pose parsing, solar-geometry calculation, scene quality control, coordinate-system resolution, point-cloud preparation, manual target annotation, per-scene geometric correction, projection, visibility screening, spectral extraction, Gaussian-support sampling, multi-view fusion, and final product export.

## Repository layout

| Path | Purpose |
|---|---|
| `config/project_config.yaml` | Main configuration file. Replace every path placeholder before running. |
| `config/crs_candidates.json` | Candidate projected coordinate reference systems. |
| `hyperspectral_pointcloud_fusion/` | Core configuration, ENVI, geometry, visibility, and PLY package. |
| `hpf3/` | Leakage-safe reconstruction and leave-one-view-out utilities. |
| `scripts/` | Numbered processing stages and optional quality-control utilities. |
| `run/run_all.bat` | Runs the complete workflow on Windows. |
| `run/run_from_10.bat` | Resumes from visibility selection after calibration and projection have been completed. |
| `requirements.txt` | Pinned Python dependencies used by this repository. |

## Software requirements

- Windows 10 or Windows 11
- Python 3.10
- A Python installation with GUI support for the manual annotation stage
- Sufficient memory and disk space for the input point cloud, ENVI cubes, intermediate projection arrays, and directional BRF products

Install the pinned dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The batch launchers search for Python in the following order:

1. the executable specified by the `HPCF_PYTHON` environment variable;
2. `<PROJECT_ROOT>\.venv\Scripts\python.exe`;
3. `<PROJECT_ROOT>\venv\Scripts\python.exe`;
4. `py.exe`; and
5. `python.exe` available on `PATH`.

## Configuration

Edit `config/project_config.yaml` before the first run. At minimum, replace these placeholders:

| Placeholder | Required value |
|---|---|
| `<PATH_TO_PYTHON_EXECUTABLE>` | Python interpreter used for this project. |
| `<PATH_TO_PROJECT_ROOT>` | Absolute path to this repository. |
| `<PATH_TO_OUTPUT_BASE>` | Parent directory for all generated products. |
| `<PATH_TO_REF_DIRECTORY>` | Directory containing the calibrated ENVI reflectance cubes and headers. |
| `<PATH_TO_BRDF_ANGLE_TABLE>` | Spreadsheet containing camera position, view geometry, and acquisition time. |
| `<PATH_TO_BREF_ANGLE_TABLE>` | Spreadsheet containing target or auxiliary angle information. |
| `<PATH_TO_POINTCLOUD_PLY>` | Input point-cloud file in PLY format. |

Do not commit local input paths, raw imagery, point clouds, calibration caches, or generated outputs to a public repository.

## Input data

The workflow expects:

- ENVI hyperspectral reflectance cubes with readable `.hdr` metadata;
- a BRDF pose table containing scene labels, camera coordinates, view geometry, and acquisition time;
- a BREF or auxiliary target-geometry table;
- a PLY point cloud whose coordinate system can be matched to one of the configured CRS candidates; and
- consistent scene identifiers across the imagery and pose tables.

The default camera model assumes 1,886 x 1,886-pixel images and horizontal and vertical fields of view of 35 degrees. Update the camera settings in the configuration if a different sensor geometry is used.

## Band modes and output roots

The run name is derived from the input point-cloud stem and `bands.mode`:

| `bands.mode` | Run directory below `<PATH_TO_OUTPUT_BASE>` | Final fused point cloud |
|---|---|---|
| `subset` | `<POINTCLOUD_STEM>` | `hpf_<POINTCLOUD_STEM>.ply` |
| `all` | `<POINTCLOUD_STEM>_allbands` | `hpf_<POINTCLOUD_STEM>_allbands.ply` |

The main run directory contains:

| Directory or file | Contents |
|---|---|
| `fusion/` | Fused spectra, band metadata, weights, and retained quality arrays. |
| `brf/` | Point-view geometry, Gaussian-support directional spectra, and BRF metadata. |
| `projection/corrected/` | Corrected point projections and retained observation records. |
| `calibration/` | Manual target annotations and per-scene correction results. |
| `source/` | Scene database, original ENVI references, point indices, coordinates, and normals required for reproducibility. |
| `product_manifest.json` | Product paths, shapes, spatial-support metadata, and method version. |
| `output_manifest.json` | Output inventory and cleanup record. |

Large previews, duplicate summaries, temporary memmaps, and products that can be regenerated from the retained core arrays are not part of the final deliverables.

## Processing stages

| Stage | Script | Main operation |
|---:|---|---|
| 00 | `00_build_scene_manifest.py` | Inventory ENVI scenes and wavelengths. |
| 01 | `01_parse_pose_tables.py` | Parse pose and angle tables. |
| 02 | `02_compute_solar_geometry.py` | Calculate solar geometry from acquisition time and camera location. |
| 03 | `03_apply_scene_quality_filter.py` | Apply scene-level completeness and quality checks. |
| 04 | `04_resolve_crs_and_estimate_hemi_center.py` | Resolve the projected CRS and estimate the scene centre. |
| 05 | `05_build_scene_database.py` | Build the per-scene database and previews. |
| 06 | `06_prepare_pointcloud.py` | Prepare point coordinates, local coordinates, and normals. |
| 07 | `07_annotate_target_in_scenes.py` | Collect or reuse manual target annotations. |
| 08 | `08_solve_per_scene_correction.py` | Estimate and apply per-scene orientation corrections. |
| 09 | `09_project_topk_candidates.py` | Project candidate point-view observations. |
| 10 | `10_select_visible_observations.py` | Screen visibility and observation quality. |
| 11 | `11_extract_and_fuse_spectra.py` | Extract spectra and build the initial weighted product. |
| 12 | `12_build_point_directional_brf_observations.py` | Build the all-view directional BRF geometry table. |
| 12b | `12b_build_gauss_support_products.py` | Rebuild all-view and weighted products with 8-pixel Gaussian support. |
| 13 | `13_export_products.py` | Export the final fused point cloud and manifests. |

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/shanhai998/hyperspectral-pointcloud-fusion/blob/main/LICENSE) for details.
