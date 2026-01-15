# how to run openevolve with blis??

- first install requirements
```bash
cd openevolve
pip install -e ".[dev]"
```

- define api key like `export OPENAI_API_KEY=<LiteLLM key>`


- then run openevolve with blis simulator.. Which currently finds RoundRobin LB
```bash
python openevolve/openevolve-run.py \
  initial_program.py \
  blis_evaluator.py \
  --config config.yaml \
  --iterations 20
```

# how to Vidur interface?

- first clone Vidur:

```bash
git clone git@github.com:microsoft/vidur.git
```

- install Vidur:

```bash
cd vidur
python3.10 -m venv .venv
source .venv/bin/activate
pip install transformers
python -m pip install -r requirements.txt
```

- run Vidur python interface

```bash
python vidur_evaluator.py
```