### Testing/Evaluation

##### Install dependencies
```bash
!pip install -U latex2sympy2 pebble sympy word2number
```

##### Base model benchmark with all samples (quick test with 256 batch size)
- Takes about 17 mins on L4 22Gb GPU

```bash
!python /content/test_sft.py --model Qwen/Qwen2.5-0.5B-Instruct --all --batch-size 256
```

##### Fine-tuned model evaluation
- Adjust `--model-path` to your fine-tuned model directory
```bash
!python test_gsm8k.py --model_path "./qwen2.5-0.5b-gsm8k-unsloth" --max_samples 2000 --batch-size 256
```
