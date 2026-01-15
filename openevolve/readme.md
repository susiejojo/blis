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