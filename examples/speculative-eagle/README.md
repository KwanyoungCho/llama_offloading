# llama.cpp/examples/speculative-eagle

build/bin/llama-speculative-eagle -m /data/youngmin/models/vicuna.gguf -md /data/youngmin/models/EAGLE.gguf -f test.txt -c 0 --color --sampling-seq k --top-k 2 -fa --temp 0.0 --draft-max 10 --draft-min 1 --draft-p-split 0.1 --n-predict 200 -ngl 40 -ngld 20 -np 10 -s 1234
