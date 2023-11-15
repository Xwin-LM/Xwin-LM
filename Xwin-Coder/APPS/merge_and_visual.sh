
# subset name: one of introductory, interview and competition
SUBSET="introductory"  # "competition" "interview" "introductory"
output_path=preds/APPS_$SUBSET


# for single node 
python merge.py --root $output_path
python test_one_solution.py -t ${SUBSET}_test.json --save $output_path --print_results
#---------------------------------------------------------------------------------------------------

# for multiple nodes 
# ROOT=/mnt/miaosen/xwin_coder_eval/APPS
# echo $MASTER_ADDR  # need to set in environment

# if [ "$NODE_RANK" -eq 0 ]  
# then  
#   sleep 15
#   python merge.py --root $output_path
#   python test_one_solution.py -t ${SUBSET}_test.json --save $output_path --print_results
# else  
#   scp -r $output_path/* $MASTER_ADDR:$ROOT/$output_path/
# fi  

