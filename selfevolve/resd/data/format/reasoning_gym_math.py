import argparse
import random
import pandas as pd
from pathlib import Path
from reasoning_gym.composite import DatasetSpec
import reasoning_gym

def create_fr(total_num):
    specs = [
        DatasetSpec(name='base_conversion', weight=1, config={}),  # default config
        DatasetSpec(name='count_bits', weight=1, config={}),  
        DatasetSpec(name='count_primes', weight=1, config={}), 
    ]
    
    rg_datas = reasoning_gym.create_dataset('composite', size=total_num, seed=42, datasets=specs)
    return rg_datas

def map_fn(example, idx, split):
    instruction_following = 'Please reason step by step and put your final answer within \\boxed{}.'
    return {
        'data_source': 'reasoning_gym',
        'prompt': [
            {
                "role": 'user',
                'content': example['question'] + instruction_following,
            }
        ],
        'ability': 'math',
        'reward_model': {'style': 'math', 'ground_truth': example['answer']},
        'extra_info': {'split': split, 'index': idx}
    }

def write_parquet(file_path, data):
    valid_data = [d for d in data if d is not None]
    if not valid_data:
        print("No data to write.")
        return
        
    df = pd.DataFrame(valid_data)
    
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(file_path, index=False)
    print(f"Successfully saved {len(df)} records to {file_path}")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Generate and split reasoning_gym datasets.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save train.parquet and test.parquet")
    parser.add_argument("--total_num", type=int, default=900, 
                        help="Total number of dataset examples to generate")
    parser.add_argument("--train_test_ratio", type=float, default=0.6666, 
                        help="Ratio of training data (e.g., 0.8 for 80% train, 20% test)")
    
    args = parser.parse_args()

    
    rg_datas = list(create_fr(args.total_num))
    random.shuffle(rg_datas)

   
    train_size = int(args.total_num * args.train_test_ratio)
    
   
    train_datas = [map_fn(example, idx, 'train') for idx, example in enumerate(rg_datas[:train_size])]
    test_datas = [map_fn(example, idx, 'test') for idx, example in enumerate(rg_datas[train_size:])]

    
    output_dir = Path(args.output_dir)
    train_path = output_dir / 'train.parquet'
    test_path = output_dir / 'test.parquet'

    
    write_parquet(train_path, train_datas)
    write_parquet(test_path, test_datas)
