# Kaggle Notebooks (No SSH Required)

Single-environment design:
- canonical repo path: `/kaggle/work_here/yolov13`
- canonical venv path: `/kaggle/work_here/yolov13/.venv`
- outputs: `/kaggle/working`

Notebooks:
- `01_train.ipynb`
- `02_validate.ipynb`
- `03_export.ipynb`
- `04_tracking.ipynb`
- `05_test.ipynb`

Optional flash backend flags:
- `Y13_INSTALL_TURING_FLASH=1` build/install Turing flash extension
- `Y13_USE_TURING_FLASH=0` disable turing flash backend at runtime (default is enabled)
- `Y13_DISABLE_FLASH=1` force fallback backend
