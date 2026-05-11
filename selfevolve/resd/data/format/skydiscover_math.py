import os
import json
import yaml
import pandas as pd
from pathlib import Path

def get_datasource():
    path_str='./selfevolve/resd/feedback/skydiscover_math'
    path_obj=Path(path_str)

    folder_names=[f.name for f in path_obj.iterdir() if f.is_dir()]
    return folder_names

USER_INSTRUCTION = """
Improve the program below for this benchmark. Respond with a complete Python solution inside a ```python``` code block.

Rules:
- Keep the evaluator entry point required for this task (e.g. `run_packing()`, `circle_packing21()`, `run()`, etc.).
- Prefer editing the EVOLVE-BLOCK region when the starter separates it.
- Keep required imports (e.g. numpy).
- Do not add imports for matplotlib, plotting, or GUI at module level; evaluation runs in a headless subprocess without optional viz packages. If the starter nests visualization inside a function, keep it lazy-imported only inside that function and never call it from top-level code.


--- initial_program.py (starter) ---
"""

DATA_TYPES=['heilbronn_triangle', 'circle_packing', 'first_autocorr_ineq', 'second_autocorr_ineq', 'third_autocorr_ineq',
              'uncertainty_ineq','sums_diffs_finite_sets', 'minimizing_max_min_dist_2','minimizing_max_min_dist_3', 'circle_packing_rect', 
              'hexagon_packing_11','hexagon_packing_12', 'matmul', 'erdos_min_overlap', 'heilbronn_convex_13','heilbronn_convex_14', 'signal_processing']
BASE_DIR = './selfevolve/resd/feedback/skydiscover_math'
CONFIG_PATH = BASE_DIR + '/{data_type}/config.yaml'
INITIAL_PATH = BASE_DIR + '/{data_type}/initial_program.py'
OUT_DIR='./selfevolve/resd/datasets/skydiscover_math/train.parquet'

def make_map_fn(data_type,index):
    cfg_path_str = CONFIG_PATH.format(data_type=data_type)
    init_path_str = INITIAL_PATH.format(data_type=data_type)
    
    cfg_path = Path(cfg_path_str)
    init_path = Path(init_path_str)

    with open(cfg_path,'r') as f:
        config=yaml.safe_load(f)
    
    starter_code = init_path.read_text(encoding="utf-8")

    user_content = USER_INSTRUCTION + "```python\n" + starter_code + "\n```"
    sys_prompt=config['prompt']['system_message']
    message=[{'role':'system','content':sys_prompt},{'role':'user','content':user_content}]
    # RLHFDataset + SDPO expect `extra_info` to be a dict. SDPO also requires each
    # `extra_info` to contain a stable ``index`` (see ray_trainer._maybe_build_self_distillation_batch).
    extra_info = {"split": "train", "bench_mark": data_type, "index": index}

    return {
        "data_source": "skydiscover_math",
        "prompt": message,
        "reward_model": {
            "style": data_type,
            "ground_truth": json.dumps({"bench_mark": data_type}),
        },
        "extra_info": extra_info,
    }

def write_parquet(file_path,data):
    valid_data = [d for d in data if d is not None]
    if not valid_data:
        print("No data to write.")
        return
        
    df = pd.DataFrame(valid_data)
    
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(file_path, index=False)
    print(f"Successfully saved {len(df)} records to {file_path}")


def run_processing():
    datas = []
    for index, data_type in enumerate(DATA_TYPES):
        datas.append(make_map_fn(data_type, index=index))

    write_parquet(OUT_DIR,datas)

if __name__=='__main__':
    run_processing()