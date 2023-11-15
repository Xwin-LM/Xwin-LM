

CKPT=<model name or path>
ID=<model id name>

python gen_model_answer.py --model-path $CKPT --model-id $ID

export OPENAI_VERSION=<api version>
export OPENAI_ENDPOINT=<endpoint>
export OPENAI_KEY=<key>

python gen_judgement.py --model-list $ID --parallel 4

python show_result.py --model-list $ID