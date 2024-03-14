sbatch main.py --data-name dsads --shuffle --miss-rate 0.6  --model-type shaspec --seed 2 --root-path ../../datasets 
sbatch main.py --data-name dsads --shuffle --miss-rate 0.6  --model-type shaspec --seed 2 --root-path ../../datasets --ablate-shared-encoder

sbatch main.py --data-name dsads --shuffle --miss-rate 0.0  --model-type shaspec --seed 2 --root-path ../../datasets 


# With weight decay
sbatch main.py --data-name dsads --shuffle --miss-rate 0.0  --model-type shaspec --seed 3 --root-path ../../datasets 
sbatch xmain.py --data-name dsads --shuffle --miss-rate 0.6  --model-type shaspec --seed 3 --root-path ../../datasets 

# Without weight decay
sbatch main.py --data-name dsads --shuffle --miss-rate 0.0  --model-type shaspec --seed 4 --root-path ../../datasets 
sbatch main.py --data-name dsads --shuffle --miss-rate 0.6  --model-type shaspec --seed 4 --root-path ../../datasets 
# --> 