sbatch main.py --data-name oppo --shuffle --model-type deepconvlstm_attn --seed 2 --difference --filtering --mixup-argmax --root-path ../../../datasets --filter-scaling-factor 1.0 --mixup-probability 0.5 --random-augmentation-prob 0.5
sbatch main.py --data-name rw --shuffle --model-type deepconvlstm_attn --seed 2 --difference --filtering --mixup-argmax --root-path ../../../datasets --filter-scaling-factor 1.0 --mixup-probability 0.5 --random-augmentation-prob 0.5
sbatch main.py --data-name rw --shuffle --model-type deepconvlstm_attn --seed 2 --difference --filtering --mixup-argmax --root-path ../../../datasets --filter-scaling-factor 1.0 --mixup-probability 1.0 --random-augmentation-prob 0.5
sbatch main.py --data-name rw --shuffle --model-type deepconvlstm --seed 2 --difference --filtering --mixup-argmax --root-path ../../../datasets --filter-scaling-factor 1.0 --mixup-probability 1.0 --random-augmentation-prob 0.5
sbatch main.py --data-name rw --shuffle --model-type deepconvlstm --seed 2 --difference --filtering --mixup-argmax --root-path ../../../datasets --filter-scaling-factor 1.0 --mixup-probability 0.5 --random-augmentation-prob 0.5
sbatch main.py --data-name rw --shuffle --model-type attend --seed 2 --difference --filtering --mixup-argmax --root-path ../../../datasets --filter-scaling-factor 1.0 --mixup-probability 0.5 --random-augmentation-prob 0.5
sbatch main.py --data-name rw --shuffle --model-type attend --seed 2 --difference --filtering --mixup-argmax --root-path ../../../datasets --filter-scaling-factor 1.0 --mixup-probability 1.0 --random-augmentation-prob 0.5
sbatch main.py --data-name rw --shuffle --model-type attend --seed 2 --difference --filtering --mixup-argmax --root-path ../../../datasets --filter-scaling-factor 1.0 --mixup-probability 0.5 --random-augmentation-prob 1.0
sbatch main.py --data-name rw --shuffle --model-type mcnn --seed 2 --difference --filtering --mixup-argmax --root-path ../../../datasets --filter-scaling-factor 1.0 --mixup-probability 0.5 --random-augmentation-prob 0.5
sbatch main.py --data-name rw --shuffle --model-type tinyhar --seed 2 --difference --filtering --mixup-argmax --root-path ../../../datasets --filter-scaling-factor 1.0 --mixup-probability 0.5 --random-augmentation-prob 0.5



python main.py --data-name bosch --shuffle --model-type tinyhar --seed 2 --difference --filtering --mixup-argmax --root-path ../../../datasets --filter-scaling-factor 1.0 --mixup-probability 0.5 --random-augmentation-prob 0.5