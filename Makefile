PY := .venv/bin/python
# xgboost wheels on macOS need libomp; reuse the copy bundled inside the scikit-learn wheel
export DYLD_LIBRARY_PATH := $(CURDIR)/.venv/lib/python3.12/site-packages/sklearn/.dylibs

.PHONY: setup train report test anatomy web demo build film stills

setup:
	~/.local/bin/uv venv --python 3.12 .venv
	~/.local/bin/uv pip install --python $(PY) -r requirements.txt
	cd web && npm install

anatomy:
	$(PY) ml/build_anatomy.py

train:
	$(PY) -m ml.train

report:
	$(PY) -m ml.report --web
	node docs/render_pdf.mjs

test:
	$(PY) -m pytest -q tests
	cd web && npm test

demo web:
	cd web && npm run dev

build:
	cd web && npm run build

film:
	node output/record.mjs

stills:
	node output/stills.mjs
