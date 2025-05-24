
from typing import Any, Dict
import os 

def parse_system_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the given configuration dictionary, construct derived parameters and file paths for the experiment,
    and merge these with the original configuration.

    This function extracts relevant values from the input dictionary (using defaults where needed), builds
    additional derived parameters (such as formatted sub-experiment names and file paths), and returns a
    consolidated dictionary containing both the original and the new values.

    Args:
        config (Dict[str, Any]): A dictionary containing configuration parameters. Expected keys include:
            - "source_type": The identifier for the source data.
            - "run_name": The experiment's name.
            - "top_k": Number of top matches.
            - "total_skus_to_map": Total SKUs to map.
            - "top_k_levenshtein": Top k Levenshtein threshold (typically 0 if feature flag is off).
            - "model_choice": Model choice parameter.
            - "sub_run_name": Base name for the sub-experiment.
            - "embedding_model_name": (Optional) Name of the embedding model (default "all-MiniLM-L6-v2").
            - "fuzzy_mode": (Optional) Fuzzy matching mode (default "levenshtein-jaccard-equal").
            - "eval_mode": (Optional) Evaluation mode (default "inference").
            - "gcp_folder_path": (Optional) GCP folder path (default "AIProductMapper/matching").
            - "size_filter_on": (Optional) Boolean flag for size filtering (default False).

    Returns:
        Dict[str, Any]: A merged dictionary containing all original configuration values plus the following derived keys:
            - source_type: As provided.
            - run_name: As provided.
            - top_k: As provided.
            - total_skus_to_map: As provided.
            - top_k_levenshtein: As provided.
            - model_choice: As provided.
            - sub_run_name: Constructed as "{sub_run_name}-{top_k}".
            - embedding_model: Resolved embedding model name.
            - fuzzy_mode: Resolved fuzzy matching mode.
            - eval_mode: Resolved evaluation mode.
            - gcp_folder_path: Resolved GCP folder path.
            - intermediate_filepath: "intermediate-outputs/{run_name}/{sub_run_name}.csv"
            - final_path: "experiments/{run_name}/{sub_run_name}_final.csv"
            - inference_df_path: "AIProductMapper/intermediate-outputs/{sub_run_name}/{sub_run_name}_{source_type}_inference_df.csv"
            - post_inference_df_path: "AIProductMapper/intermediate-outputs/{sub_run_name}/{sub_run_name}_{source_type}_post_inference_df.csv"
            - retrieval_df_path: "AIProductMapper/intermediate-outputs/{sub_run_name}/{sub_run_name}_{source_type}_retrieval_df.csv"
            - gcp_final_path: "AIProductMapper/intermediate-outputs/{sub_run_name}/{sub_run_name}_{source_type}_final.csv"
            - size_filter_on: Boolean flag for size filtering.

    """
    # Extract values with defaults.
    source_type = os.getenv("SOURCE_TYPE",config.get("source_type"))
    run_name = os.getenv("RUN_NAME",config.get("run_name"))
    top_k = os.getenv("TOP_K",config.get("top_k"))
    total_skus_to_map = os.getenv("TOTAL_SKUS_TO_MAP",config.get("total_skus_to_map"))
    top_k_levenshtein = os.getenv("TOP_K_LEVENSHTEIN",config.get("top_k_levenshtein"))
    model_choice = config.get("model_choice")
    
    base_sub_run_name = os.getenv("SUB_RUN_NAME",config.get("sub_run_name"))
    sub_run_name = f"{base_sub_run_name}-{top_k}"
    
    source_data_gcp_path = os.getenv("SOURCE_DATA_GCP_PATH",config.get("source_data_gcp_path"))
    intermediate_filepath = f"intermediate-outputs/{run_name}/{sub_run_name}.csv"
    final_path = f"experiments/{run_name}/{sub_run_name}_final.csv"
    
    embedding_model = config.get("embedding_model_name", "all-MiniLM-L6-v2")
    fuzzy_mode = config.get("fuzzy_mode", "levenshtein-jaccard-equal")
    eval_mode = config.get("eval_mode", "inference")
    gcp_folder_path = config.get("gcp_folder_path", "AIProductMapper/matching")
    size_filter_on = config.get("size_filter_on", False)
    data_load_method = config.get("data_load_method", "snowflake")
    skip_abb_conversion = config.get("skip_abb_conversion", False)
    abb_path = config.get("abb_path", "AIProductMapper/abbreviations/abb_matches_initials.csv")

    base_intermediate_path = f"AIProductMapper/intermediate-outputs/{run_name}_{sub_run_name}_{source_type}"
    inference_df_path = f"{base_intermediate_path}/{sub_run_name}_{source_type}_inference_df.csv"
    post_inference_df_path = f"{base_intermediate_path}/{sub_run_name}_{source_type}_post_inference_df.csv"
    retrieval_df_path = f"{base_intermediate_path}/{sub_run_name}_{source_type}_retrieval_df.csv"
    gcp_final_path = f"{base_intermediate_path}/{sub_run_name}_{source_type}_final.csv"
    gcp_aligned_final_path = f"{base_intermediate_path}/{sub_run_name}_{source_type}_aligned_final.csv"
    post_cross_reference_df_path = f"{base_intermediate_path}/{sub_run_name}_{source_type}_post_cross_reference_df.csv"
    pure_ai_match_df_path = f"{base_intermediate_path}/{sub_run_name}_{source_type}_pure_ai_match_df.csv"


    # Merge original config with derived values.
    new_config: Dict[str, Any] = dict(config)
    new_config.update({
        "source_type": source_type,
        "run_name": run_name,
        "top_k": int(top_k),
        "total_skus_to_map": total_skus_to_map,
        "top_k_levenshtein": top_k_levenshtein,
        "model_choice": model_choice,
        "sub_run_name": sub_run_name,
        "embedding_model": embedding_model,
        "fuzzy_mode": fuzzy_mode,
        "eval_mode": eval_mode,
        "gcp_folder_path": gcp_folder_path,
        "intermediate_filepath": intermediate_filepath,
        "data_load_method": data_load_method,
        "skip_abb_conversion":skip_abb_conversion,
        "abb_path":abb_path,
        "final_path": final_path,
        "inference_df_path": inference_df_path,
        "post_inference_df_path": post_inference_df_path,
        "retrieval_df_path": retrieval_df_path,
        "gcp_final_path": gcp_final_path,
        "size_filter_on": size_filter_on,
        "gcp_aligned_final_path": gcp_aligned_final_path,
        "source_data_gcp_path": source_data_gcp_path,
        "post_cross_reference_df_path": post_cross_reference_df_path,
        "pure_ai_match_df_path": pure_ai_match_df_path
    })
    
    return new_config
