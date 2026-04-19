python main.py \
  --model_name ./models/pythia-70m \
  --method streaming \
  --dataset wikitext \
  --budget 256 \
  --context_length 2000 \
  --max_new_tokens 50