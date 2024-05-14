## Intro

This repo is used for decoder-only LMs' inference and the code is based on transformers

- Support batch_generate
- Return scores/probs for the generated tokens
- Supporting datasets: Natural Questions; MMLU

## Example

You can try using:

```bash
bash run_nq.sh
bash run_mmlu.sh
```

## Note



> `llm_deepspeed.py` is a demo for using distributed inference
>
> `generation_utils.py` is the generation file in transformers used for learning
