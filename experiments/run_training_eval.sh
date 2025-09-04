#!/bin/bash

# Run PointLSTM training and evaluation
python compute_dataset_stats.py && python main.py --config pointlstm.yaml --device 0 --eval-interval 10 --num-epoch 100 && python main.py --config pointlstm.yaml --device 0 --eval-interval 1 --weights work_dir/baseline/epoch100_model.pt