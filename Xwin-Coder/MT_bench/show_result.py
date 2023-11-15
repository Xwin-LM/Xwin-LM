"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
from pathlib import Path
import pandas as pd

def sort_dataframe_by_question_and_judge(df):
    """
    Sorts a DataFrame by 'question_id' and 'judge' columns without modifying the original DataFrame.
    
    Parameters:
        df (DataFrame): The DataFrame to sort.

    Returns:
        DataFrame: The sorted DataFrame.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Convert 'judge' to a sortable string
    df_copy['judge_str'] = df_copy['judge'].astype(str)
    
    # Sort by 'question_id' and 'judge_str'
    df_copy.sort_values(['question_id', 'judge_str'], inplace=True)
    
    # Optionally, convert 'judge_str' back to list
    df_copy['judge'] = df_copy['judge_str'].apply(eval)
    df_copy.drop('judge_str', axis=1, inplace=True)
    
    return df_copy

def display_result_single(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    # TODO: we only deal with single model case for now
    evaluation_df = pd.read_json(input_file, lines=True)

    # sort the evaluation_df according to question_id
    evaluation_df = sort_dataframe_by_question_and_judge(evaluation_df)
    # save back the sorted evaluation_df into the input_file
    evaluation_df.to_json(input_file, orient='records', lines=True)

    evaluation_df = evaluation_df[evaluation_df['score']!=-1]
    if args.model_list is not None:
        assert len(args.model_list) == 1
        md = args.model_list[0]
        evaluation_df = evaluation_df[evaluation_df["model"]==md]
    print(f"Total num of samples: {len(evaluation_df)}")
    questions_df = pd.read_json(Path(__file__).parent/"data"/"mt_bench"/"question.jsonl",lines=True)
    # Calculating the score per turn per category
    score_per_category_df = pd.merge(evaluation_df, questions_df[['question_id', 'category']], on='question_id')
    score_per_category_df = score_per_category_df.groupby(['turn', 'category'])['score'].mean().reset_index()

    # Pivoting the DataFrame for readability
    readable_format_df = score_per_category_df.pivot(index='turn', columns='category', values='score')
    readable_format_df.fillna(0, inplace=True)
    
    # Calculating the average score per turn
    average_score_per_turn_series = evaluation_df.groupby('turn')['score'].mean()
    
    # Adding the average score per turn as a new column
    readable_format_df['average_score'] = average_score_per_turn_series
    
    # Calculating the average score for each category across all turns
    category_average_score = readable_format_df.drop(columns=['average_score']).mean()
    
    # Adding the average score for each category as a new row
    readable_format_df.loc['cat_average'] = category_average_score
    readable_format_df.loc['cat_average', 'average_score'] = category_average_score.mean()
    # Reordering the columns to place "average_score" at the leftmost position
    cols = ['average_score'] + [col for col in readable_format_df.columns if col != 'average_score']
    readable_format_df = readable_format_df[cols]

    print(readable_format_df)
    
    # return readable_format_df

    """original code
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]
    print(f"Total number of samples: {len(df)}")

    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    if args.bench_name == "mt_bench":
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        print(df_2.sort_values(by="score", ascending=False))

        print("\n########## Average ##########")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        print(df_3.sort_values(by="score", ascending=False))

    """


def display_result_pairwise(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df_all = df_all[(df_all["g1_winner"] != "error") & (df_all["g2_winner"] != "error")]

    model_list = (
        df_all["model_1"].unique().tolist() + df_all["model_2"].unique().tolist()
    )
    model_list = list(set(model_list))

    list_res = []
    # traverse df row by row
    for index, row in df_all.iterrows():
        if args.model_list is not None and row["model_1"] not in args.model_list:
            continue
        if args.baseline_model is not None:
            if args.baseline_model not in [row["model_1"], row["model_2"]]:
                continue
        if row["g1_winner"] == "tie" or row["g1_winner"] != row["g2_winner"]:
            list_res.append({"model": row["model_1"], "win": 0, "loss": 0, "tie": 1})
            list_res.append({"model": row["model_2"], "win": 0, "loss": 0, "tie": 1})
        else:
            if row["g1_winner"] == "model_1":
                winner = row["model_1"]
                loser = row["model_2"]
            else:
                winner = row["model_2"]
                loser = row["model_1"]
            list_res.append({"model": winner, "win": 1, "loss": 0, "tie": 0})
            list_res.append({"model": loser, "win": 0, "loss": 1, "tie": 0})

    df = pd.DataFrame(list_res)
    df = df.groupby(["model"]).sum()

    # remove baseline model
    if args.baseline_model is not None:
        df = df[df.index != args.baseline_model]
    # add win rate
    df["win_rate"] = df["win"] / (df["win"] + df["loss"] + df["tie"])
    df["loss_rate"] = df["loss"] / (df["win"] + df["loss"] + df["tie"])
    # each tie counts as 0.5 win + 0.5 loss
    df["win_rate_adjusted"] = (df["win"] + 0.5 * df["tie"]) / (
        df["win"] + df["loss"] + df["tie"]
    )
    # print(df.sort_values(by="win_rate", ascending=False))
    # print(df.sort_values(by="loss_rate", ascending=True))
    print(df.sort_values(by="win_rate_adjusted", ascending=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    args = parser.parse_args()

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")
    display_result_func(args)