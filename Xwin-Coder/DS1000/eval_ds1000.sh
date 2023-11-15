
for SUBSET_NAME in "Pandas" "Numpy" "Pytorch" "Scipy" "Sklearn" "Tensorflow" "Matplotlib"
do

    # temperature, if using greedy decode, temp=0.0
    temp=0.0

    # Number of predictions, for fast greedy_decode setting, set to 1, for standard setting, set to 200
    pred_num=1
    # generation batch size, set it according to your GPU memory, but smaller than pred_num
    num_seqs_per_iter=1

    # temperary generation path
    # output_path=preds/DS1000_T${temp}_N${pred_num}_SUB${SUBSET_NAME}
    output_path=preds/DS1000_T${temp}_N${pred_num}_SUB${SUBSET_NAME}

    save_path=../generation_for_harness/ds1000_T${temp}_N${pred_num}_SUB${SUBSET_NAME}
    mkdir -p ../generation_for_harness

    echo 'Output path: '$output_path
    python process_ds1000.py --path ${output_path} --out_path ${save_path}.jsonl --add_prompt --subset $SUBSET_NAME

done

