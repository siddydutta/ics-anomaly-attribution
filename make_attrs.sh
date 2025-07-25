#!/bin/bash
cd explain-eval-attacks/

for ATTACK in {0..31}
do
    echo "Processing ATTACK $ATTACK..."

    # SM attribution
    FILE_SM="../explanations-dir/explain23-pkl/explanations-saliency_map_mse_history-LSTM-SWAT-l2-hist50-units64-results-${ATTACK}-true150.pkl"
    if [ ! -f "$FILE_SM" ]; then
        echo "  Generating SM attributions..."
        python main_grad_explain_attacks.py LSTM SWAT $ATTACK --explain_params_methods SM --run_name results --num_samples 150
    else
        echo "  SM file exists, skipping."
    fi

    # LEMNA attribution
    FILE_LEMNA="../explanations-dir/explain23-pkl/explanations-LEMNA-LSTM-SWAT-l2-hist50-units64-results-${ATTACK}-true150.pkl"
    if [ ! -f "$FILE_LEMNA" ]; then
        echo "  Generating LEMNA attributions..."
        python main_bbox_explain_attacks.py LSTM SWAT $ATTACK --explain_params_methods LEMNA --run_name results --num_samples 150
    else
        echo "  LEMNA file exists, skipping."
    fi

    # Ensemble attribution
    FILE_ENSEMBLE="../explanations-dir/explain23-json/explanations-ensemble-LSTM-SWAT-l2-hist50-units64-results-${ATTACK}-true150.json"
    if [ ! -f "$FILE_ENSEMBLE" ]; then
        echo "  Generating Ensemble attributions..."
        cd ..
        python save_ensemble_attributions.py --attack_index $ATTACK
        cd explain-eval-attacks/
    else
        echo "  Ensemble file exists, skipping."
    fi
done
