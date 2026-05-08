# ==== Common ====
PY   := python
CODE := $(HOME)/eigenmode_fingerprints/code

# ==== Part 1: Energy spectrum & criticality ====
OUT_EN := $(HOME)/eigenmode_fingerprints/pang_out/group

group_energy: \
	$(OUT_EN)/energy_spectrum_group.csv \
	$(OUT_EN)/group_spectrum_model_fits.png \
	$(OUT_EN)/group_model_compare_loglog.png

$(OUT_EN)/energy_spectrum_group.csv:
	$(PY) $(CODE)/energy_compute_per_run.py
	$(PY) $(CODE)/energy_group_aggregate.py

$(OUT_EN)/group_spectrum_model_fits.png $(OUT_EN)/group_model_compare_loglog.png: $(OUT_EN)/energy_spectrum_group.csv
	$(PY) $(CODE)/energy_fit_criticality.py


# ==== Part 2: Sentence-boundary GLM ====
# All outputs live under pang_out/group_boundary_glm
OUT_GL := $(HOME)/eigenmode_fingerprints/pang_out/group_boundary_glm

# Individual steps you can run as needed
boundaries:
	$(PY) $(CODE)/build_sent_bound_per_run.py --unit auto

trs:
	$(PY) $(CODE)/report_trs.py

glm:
	$(PY) $(CODE)/rebuild_glm_per_run.py

group_boundary:
	$(PY) $(CODE)/group_aggregate_perrun.py

subject:
	$(PY) $(CODE)/subject_level_aggregate.py

plots:
	$(PY) $(CODE)/plot_by_wavelength.py

coverage:
	$(PY) $(CODE)/coverage_report.py

# Full Part-2 pipeline in order
boundary_glm_full: boundaries trs glm group_boundary subject plots coverage


# ==== Convenience ====
# Run *both* parts end-to-end
all: group_energy boundary_glm_full

.PHONY: group_energy boundaries trs glm group_boundary subject plots coverage boundary_glm_full all
