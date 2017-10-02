# Example of distributed tensorflow
## Latest

To train with 3 workers and 1 parameter server in four processes, run:

```bash
python run_updated.py
```

Monitor progress in another terminal with:

```bash
watch -n 2 "ps aux | grep [p]ython"
```

And remember to close the parameter server after all the workers terminate:

```bash
kill -9 PID
```

## Deprecated

```bash
python run_original.py
```
