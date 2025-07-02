#!/usr/bin/env bash

echo "🔄 Starting data loading step..."
python3 ghost_brand/data_loading.py
echo "✅ Data loading complete."

echo "🔄 Starting main pipeline step..."
python3 ghost_brand/main.py
echo "✅ Pipeline finished."

# echo "🔄 Starting training step..."
# bash training/finetune_v15.sh
# echo "✅ Training complete."
