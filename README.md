# Evaluating Population based Reinforcement Learning for Transfer Learning

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub](https://img.shields.io/github/license/Frederik-L/evaluating-population-based-reinforcement-learning-for-transfer-learning)

## Requirenments: 
- Python 3.7+
- NumPy 
- Pytorch
- tqdm
- [Gridworld](https://github.com/Frederik-L/RL-environment-Gridworld)
- TensorBoard (optional)


## Usage
Creating a trainer in the specific trainer file ```pbt_trainer.py```, it is equal for all three algorithms.
To initialize a trainer three container classes must be generated and passed as a parameter.
```
model_container = ModelContainer()
util_container = UtilityContainer()
pbt = PopulationContainer()
```
All values can be set to default except the ```environment_id``` in the utility container, it can be set as one of these.
```
util_container.environment_id = "empty-10x10-random"
util_container.environment_id = "hardcore-10x10-random"
```
Then the trainer can be created.
```
trainer = PopulationBasedTrainerReinforce(util_container, model_container, pbt)
```
### Baseline
To initialize a baseline run without transfer learning use:
```
trainer.init_single()
```
To Initiate a repeated version of the baseline with transfer learning use:
```
repetitions = 20
trainer.init_single_transfer_repeat(repetitions, "Single_transfer")
```
The parameter ```repetitions``` is the number of baseline agents.

### Population based training
To start population based training without transfer learning use:
```
trainer.init_population_training()

```
To start population based training with transfer learning use:
```
trainer.population_trainer_transfer()
```

## Known issues 
- PPO model has a chance to return ```NaN``` crashing the trainer

## List of publications:
- [Evaluating Population based Reinforcement Learning for Transfer Learning](https://github.com/Frederik-L/evaluating-population-based-reinforcement-learning-for-transfer-learning/publications/Evaluating_Population_Based_Reinforcement_Learning_for_Transfer_Learning.pdf) (2021)

Please use this bibtex if you want to cite this repository in your publications:
```
@misc{pbt-transfer,
  author = {Liebig, Jan Frederik},
  title = {Evaluating Population based Reinforcement Learning for Transfer Learning},
  year = {2021},
  howpublished = {https://github.com/Frederik-L/evaluating-population-based-reinforcement-learning-for-transfer-learning},
}
```

## References
Repositories that helped to code the algorithms 
- [Reinforce](https://github.com/goodboychan/goodboychan.github.io/blob/main/_notebooks/2020-08-06-03-Policy-Gradient-With-Gym-MiniGrid.ipynb)
- [DQN](https://github.com/pytorch/tutorials) 
- [PPO](https://github.com/nikhilbarhate99/PPO-PyTorch) 