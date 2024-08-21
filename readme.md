## Intro

This repo is used for decoder-only LMs' inference and the code is based on transformers. **We fully utilized the Transformers library without using any inference frameworks.** The repository is continuously updated.

- Support batch_generate
- Return `scores`/`hidden_states` for the generated tokens
- Supporting task format: free-form generation; multi-choice qa
- Currently, only single-GPU inference is supported.

## Usage

### Free-form Generation
```bash
python -u run_nq.py \
    --source YOUR_DATA_PATH \
    --type qa \
    --ra none \
    --outfile YOUR_OUT_FILE_PATH \
    --model_path YOUR_MODEL_PATH \
    --batch_size 36 \
    --task nq \
    --max_new_tokens 64 \
    --hidden_states 1 \
    --hidden_idx_mode every \
    --need_layers mid
```
- `--ra`: If retrieval augmentation is needed, you can specify the document type; otherwise, set it to `none`.
- `--hidden_states`: If you want to obtain the hidden state information related to the generated tokens, you need to specify this parameter; otherwise, remove it.
- `--hidden_idx_mode`: We support [`first,last,avg,min,every']
  - `first, last`: Obtain the hidden state at the specified layers for the **first** or **last** token during generation. 
  - `avg`: Obtain the average hidden state across all the generated tokens at the specified layers.
  - `min`: Obtain the hidden state of the generated token with min probability at the specified layers.
- `--need_layers`: We support [`all,last,mid`], which specify all the layers, the last layer, the mid layer (16 for 32-layer models) for getting hidden state information.


### Multi-Choice Generation

```bash
python run_mmlu.py \
    --source YOUR_DATA_PATH \
    --type qa \
    --ra none \
    --outfile YOUR_OUT_FILE_PATH \
    --n_shot 0 \
    --model_path YOUR_MODEL_PATH \
    --batch_size 16 \
    --task mmlu \
    --max_new_tokens 64 \
    --hidden_states 1 \
    --need_layers mid
```
- Very similar to `Free-Form Generation`.
- We find the choices [`A,B,C,D`] in the response and get related information, so there is no need to specify `--hidden_idx_mode`
## Note
- `--hidden_idx_mode, --need_layers` only works when you specify `--hidden_states 1`

We support only `llama2-chat` and `llama3-instruct` series models because each llm need its own prompt format. You can add more prompt templates in `utils/prompt.py` to support more LLMs.
- `llama3-8b-instruct` use `'<|eot_id|>'` instead of `<eos>` to represent the end of generation.

## Example
Free-form generation
```bash
python -u run_nq.py \
    --source ./data/nq/nq-dev.jsonl \
    --type qa \
    --ra none \
    --outfile ./data/nq/nq-dev-res.jsonl \
    --model_path YOUR_MODEL_PATH \
    --batch_size 36 \
    --task nq \
    --max_new_tokens 64 \
```

Multi-choice generation
```bash
python run_mmlu.py \
    --source ./data/truthfulqa \
    --type qa \
    --ra none \
    --outfile ./data/truthfulqa/res/ \
    --n_shot 0 \
    --model_path YOUR_MODEL_PATH \
    --batch_size 16 \
    --task tq \
    --max_new_tokens 64 \
```

## Core Files
- `run_mmlu/run_nq.py`
- `utils/prompt.py, data.py, llm.py`

> `llm_deepspeed.py` is a demo for using distributed inference
>
> `generation_utils.py` is the generation file in transformers used for learning
