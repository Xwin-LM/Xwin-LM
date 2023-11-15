
# If you first run this scrip, run the following to download data from original github repo
# wget https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz
# tar -zxvf APPS.tar.gz
# rm -f APPS.tar.gz

# set your model path that can be read by .from_pretrained() here 
# MD="../ckpts/34B_codellama/checkpoint-3400"
MD=<model name or path>
NUM_BEAMS=1
N=5
temp=0.2

# subset name: one of introductory, interview and competition

for SUBSET in "introductory" "interview" "competition"; do
  output_path=preds/APPS_$SUBSET
  # rm -rf $output_path
  mkdir -p ${output_path}
  echo 'Output path: '$output_path
  echo 'Model to eval: '$model


  #for single node evaluation
  gpu_num=4
  gpu_per_node=4
  start_gpu=0
  end_gpu=$gpu_num


  #for multi-nodes evaluation
  # gpu_per_node=8
  # gpu_num=$((gpu_per_node*WORLD_SIZE))    # WORLD_SIZE set in enviroment
  # start_gpu=$((NODE_RANK*gpu_per_node))    # NODE_RANK set in enviroment
  # end_gpu=$(( (NODE_RANK+1)*gpu_per_node ))


  for ((i = $start_gpu; i < $end_gpu; i++)); do
    start_ratio=$(echo "scale=3; $i * 1 / $gpu_num" | bc) 
    end_ratio=$(echo "scale=3; ($i+1) * 1 / $gpu_num" | bc) 

    gpu=$((i%gpu_per_node))
    echo 'Running process #' ${i} 'from' $start_ratio 'to' $end_ratio 'on GPU' ${gpu}
    (
      CUDA_VISIBLE_DEVICES=$gpu python generate_gpt_codes.py -t ${SUBSET}_test.json --save $output_path \
      --arch $MD --load $MD --num-beams $NUM_BEAMS --temperature $temp --N $N --start_ratio $start_ratio --end_ratio $end_ratio --vllm
    ) &
  done
  wait

  # Evaluate results and save result in files
  for ((i = $start_gpu; i < $end_gpu; i++)); do
    start_ratio=$(echo "scale=3; $i * 1 / $gpu_num" | bc) 
    end_ratio=$(echo "scale=3; ($i+1) * 1 / $gpu_num" | bc) 
    (
      python test_one_solution.py -t ${SUBSET}_test.json --save $output_path --start_ratio $start_ratio --end_ratio $end_ratio 
    ) &
  done
  wait
done

