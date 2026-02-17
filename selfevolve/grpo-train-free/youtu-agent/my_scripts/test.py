import sqlite3
import pandas as pd
import os

def stat1():
    db_path="/data/hal245/op_distill/youtu-agent/test.db"
    conn = sqlite3.connect(db_path)
    table_name = "evaluation_data"
    #target_exp_id = "qwen3-30b-a3b-thinking_math_AIME24_eval"
    #target_exp_id="qwen3-30b-a3b-math_AIME25_eval"
    target_exp_id="qwen3-4b_math_AIME25_eval"

    query_acc = f"SELECT reward FROM {table_name} WHERE stage='judged' AND exp_id='{target_exp_id}'"
    df_acc = pd.read_sql_query(query_acc, conn)

    if not df_acc.empty:
        total = len(df_acc)
        correct = df_acc[df_acc['reward'] == 1].shape[0]
        accuracy = correct / total
        
        print(f"\n📊 统计结果 (Exp: {target_exp_id})")
        print(f"--------------------------------")
        print(f"总样本数: {total}")
        print(f"正确样本: {correct}")
        print(f"准确率  : {accuracy:.2%}")
        print(f"--------------------------------")

if __name__=='__main__':
    stat1()