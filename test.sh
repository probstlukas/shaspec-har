sbatch main.py --data-name dsads --shuffle --miss-rate 0.6  --model-type shaspec --seed 2 --root-path ../../datasets 
sbatch main.py --data-name dsads --shuffle --miss-rate 0.6  --model-type shaspec --seed 2 --root-path ../../datasets --ablate-shared-encoder

sbatch main.py --data-name dsads --shuffle --miss-rate 0.0  --model-type shaspec --seed 2 --root-path ../../datasets 
sbatch main.py --data-name dsads --shuffle --miss-rate 0.0  --model-type shaspec --seed 3 --root-path ../../datasets --datanorm-type minmax
