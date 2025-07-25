import numpy as np
import pickle
import argparse
import json

# Import helper functions
from utils.attack_utils import get_attack_indices, is_actuator
from data_loader import load_test_data

def analyze_ideal_timing_ensemble_with_ranking(attack_index):
    """
    Replicates the 'ideal_detection_timing' analysis and ranks features based on the time-averaged ensemble scores.
    """
    # --- Configuration ---
    MODEL_NAME = "LSTM-SWAT-l2-hist50-units64-results"
    DATASET = "SWAT"
    ATTACK_INDEX = attack_index
    BETA = 2.5
    NUM_SAMPLES = 150 # Analysis window

    # --- 1. Load Data and Attack Details ---
    try:
        _, _, sensor_cols = load_test_data(DATASET)
        num_features = len(sensor_cols)
        
        attacks, true_labels = get_attack_indices(DATASET)
        attack_start_time = np.min(attacks[ATTACK_INDEX])
        true_label = true_labels[ATTACK_INDEX][0]

        all_mses = np.load(f'meta-storage/model-mses/mses-{MODEL_NAME}-ns.npy')
        sm_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-saliency_map_mse_history-{MODEL_NAME}-{ATTACK_INDEX}-true150.pkl', 'rb'))
        lemna_full = pickle.load(open(f'explanations-dir/explain23-pkl/explanations-LEMNA-{MODEL_NAME}-{ATTACK_INDEX}-true150.pkl', 'rb'))
    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e.filename}")
        return

    # --- 2. Process Attributions Across the 150-Timestep Window ---
    mse_scores_window = all_mses[attack_start_time : attack_start_time + NUM_SAMPLES]
    sm_scores_window = np.sum(np.abs(sm_full), axis=1)
    lemna_scores_window = np.sum(np.abs(lemna_full), axis=1)

    # --- 3. Calculate Ranks and Averages ---
    ensemble_time_averaged_scores = np.zeros(num_features)

    for i in range(NUM_SAMPLES):
        mse_slice = mse_scores_window[i]
        sm_slice = sm_scores_window[i]
        lemna_slice = lemna_scores_window[i]

        mse_norm = mse_slice / np.sum(mse_slice)
        sm_norm = sm_slice / np.sum(sm_slice)
        lemna_norm = lemna_slice / np.sum(lemna_slice)
        
        ensemble_scores_slice = np.zeros(num_features)
        for j in range(num_features):
            if is_actuator(DATASET, sensor_cols[j]):
                ensemble_scores_slice[j] = mse_norm[j] + BETA * sm_norm[j] + BETA * lemna_norm[j]
            else:
                ensemble_scores_slice[j] = mse_norm[j] + sm_norm[j] + lemna_norm[j]
        
        ensemble_time_averaged_scores += (ensemble_scores_slice / np.sum(ensemble_scores_slice))

    # --- 4. Generate Ranked List ---
    final_ranked_list = []
    for i in range(num_features):
        feature_name = sensor_cols[i]
        score = ensemble_time_averaged_scores[i]
        final_ranked_list.append((feature_name, score))
    # Sort by score in descending order
    final_ranked_list.sort(key=lambda x: x[1], reverse=True)

    # --- 5. Save Results as JSON ---
    json_results = {
        "attack": ATTACK_INDEX,
        "attributions": [
            {"feature": feature, "score": float(score)}
            for feature, score in final_ranked_list
        ],
        "true_label": true_label
    }
    output_path = f"explanations-dir/explain23-json/explanations-ensemble-{MODEL_NAME}-{ATTACK_INDEX}-true150.json"
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_index", type=int, required=True, help="Attack index to analyze")
    args = parser.parse_args()
    analyze_ideal_timing_ensemble_with_ranking(args.attack_index)
