
# generate and evaluate APPS in this script

# Run this to download dataset when first runing, if you failed to unzip it, you should go to the repo and download manually
# munally download from https://github.com/HKUNLP/DS-1000/blob/main/ds1000_data.zip
# unzip ds1000_data.zip
model=<model path or name>
temp=0.0
max_len=2048
pred_num=1
num_seqs_per_iter=1
gpu_num=4

for SUBSET_NAME in "Pandas" "Numpy" "Pytorch" "Scipy" "Sklearn" "Tensorflow" "Matplotlib"
do
  # temperary generation path
  output_path=preds/DS1000_T${temp}_N${pred_num}_SUB${SUBSET_NAME}
  mkdir -p ${output_path}
  echo 'Output path: '$output_path
  echo 'Model to eval: '$model


  index=0
  for ((i = 0; i < $gpu_num; i++)); do
    start_ratio=$(echo "scale=2; $i * 1 / $gpu_num" | bc) 
    end_ratio=$(echo "scale=2; ($i+1) * 1 / $gpu_num" | bc) 

    gpu=$((i))
    echo 'Running process #' ${i} 'from' $start_ratio 'to' $end_ratio 'on GPU' ${gpu}
    ((index++))
    (
      CUDA_VISIBLE_DEVICES=$gpu python ds1000_gen.py --model ${model} --subset $SUBSET_NAME \
        --start_ratio ${start_ratio} --end_ratio ${end_ratio} --temperature ${temp} \
        --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} --output_path ${output_path} --vllm
    ) &
  done
  wait
done

