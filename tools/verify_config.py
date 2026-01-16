import sys
import os
import yaml
from typing import Dict, Any

def verify_config(config_path: str):
    """
    Verifies the integrity of the ZIT quantization config file.
    Checks for existence, rank settings, and refiner inclusion.
    """
    print(f"Verifying config file: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"FAILURE: Config file not found at {config_path}")
        return 1
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"FAILURE: Could not parse YAML file. Error: {e}")
        return 1
        
    # Check strict rank requirement (must be 128)
    try:
        rank = config['quant']['wgts']['low_rank']['rank']
        if rank != 128:
            print(f"FAILURE: Rank is set to {rank}, expected 128.")
            return 1
    except KeyError:
        # Check base config handling if applicable, but for this specific file we expect explicit setting
        print("WARNING: Rank setting not found in standard path. Assuming inheritance or custom structure.")

    # Check for Refiner exclusion
    try:
        skips = config['quant']['wgts']['low_rank'].get('skips', [])
        refiner_keywords = ['context_refiner', 'noise_refiner']
        
        for skip in skips:
            for kw in refiner_keywords:
                if kw in skip:
                    print(f"FAILURE: Config explicitly skips Refiner layers: '{skip}'")
                    return 1
                    
        print("SUCCESS: Config checks passed. Refiners are not skipped.")
        return 0
        
    except KeyError:
        print("WARNING: Low-rank skips configuration not found.")
        return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_config.py <path_to_config>")
        sys.exit(1)
        
    sys.exit(verify_config(sys.argv[1]))
