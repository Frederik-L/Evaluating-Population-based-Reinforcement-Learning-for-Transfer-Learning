# @title:    pbt_trainer.py
# @author:   Jan Frederik Liebig
# @date:     07.09.2021

############################################################
# Imports
import torch
import time
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from src.utility.container import (
    ModelContainer,
    UtilityContainer,
    StatisticContainer,
    FrameType,
    PopulationContainer,
    PopulationModelContainer,
    ImageContainer,
    ProximalPolicyOptimizationContainer,
    ModelSavingContainer,
)
from tqdm import tqdm
import torchvision.transforms as transforms
from src.gridworld.gridworld import Gridworld
import logging
import torchvision
from src.gridworld_trainer.ppo.memory import MemoryPPO
from src.gridworld_trainer.ppo.model import PPONetwork, PPONetwork3D, PPOModel
import os
import copy
import torch.nn as nn
import torch.optim as optim


############################################################
# Code


class PopulationBasedTrainerPPO:
    def __init__(
            self,
            util_container: UtilityContainer,
            model_container: ModelContainer,
            pbt_container: PopulationContainer,
    ):
        """
        Initializes a trainer for training or population based training with transfer training
        @params:
            util_container  a utility container with needed information for initialization
            model_container a utility container with information for the model
            pbt_container   a utility container with information for population based training
        """
        self.model_container = model_container
        self.utility_container = util_container
        self.pbt_container = pbt_container
        self.algorithm_name = "PPO"
        self.frameType_name = ""

        self.get_state = None
        self.reset_env = None
        self.make_model = None

        self.start_time = int(time.time())

        if self.utility_container.path is None:
            self.path = "../../../logs/gridworld/ppo/logs-" + str(self.start_time)
        else:
            self.path = self.utility_container.path

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if self.utility_container.init_logging:
            logging.basicConfig(
                filename=self.path + "/info.log", filemode="w", level=logging.DEBUG
            )

        if model_container.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                if self.utility_container.logging:
                    logging.info("Running on the GPU")
            else:
                self.device = torch.device("cpu")
                if self.utility_container.logging:
                    logging.info("Running on the CPU")
            model_container.device = self.device

        if self.utility_container.seed is None:
            self.seed = self.start_time
            self.utility_container.seed = self.seed
        else:
            self.seed = self.utility_container.seed

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.utility_container.logging:
            logging.info(f"Seed: {self.seed}")

        self.transformer = transforms.ToTensor()

        self.env = Gridworld.make(self.utility_container.environment_id)

        self.obs_size = self.env.obs_size
        self.num_actions = self.env.n_actions

        self.model_container.num_inputs_x = self.obs_size
        self.model_container.num_inputs_y = self.obs_size
        self.model_container.num_outputs = self.num_actions

        self.init_frame_type()
        self.obs, _ = self.reset_env()

    def init_frame_type(self):
        """
        Initializes several functions depending on the frame type
        """
        if self.utility_container.frame_type == FrameType.Single:
            self.frameType_name = "single_frame"
            self.get_state = self.get_state_single_frame
            self.reset_env = self.reset_env_single_frame
            self.model_container.num_channel = 3
            self.model_container.num_images = 1
            self.make_model = self.make_model_2d

        if self.utility_container.frame_type == FrameType.Stacked2D:
            self.frameType_name = "2D_stacked_frames"
            self.get_state = self.get_state_2d_stacked_frame
            self.reset_env = self.reset_env_2d_stacked_frame
            self.model_container.num_channel = 9
            self.model_container.num_images = 1
            self.make_model = self.make_model_2d

        if self.utility_container.frame_type == FrameType.Stacked3D:
            self.frameType_name = "3D_stacked_frames"
            self.get_state = self.get_state_3d_stacked_frame
            self.reset_env = self.reset_env_3d_stacked_frame
            self.model_container.num_channel = 3
            self.model_container.num_images = 3
            self.make_model = self.make_model

    def make_memory(self):
        """
        Generates a memory for the ppo algorithm
        @return:
            the generated memory
        """
        memory = MemoryPPO()
        return memory

    def make_model_2d(self):
        """
        Method for generating a mode for 2d stacked frames or a single frame
        @return
            the generated model
        """
        actor = PPONetwork(self.model_container, self.obs)
        actor_old = PPONetwork(self.model_container, self.obs)
        actor_old.load_state_dict(actor.state_dict())
        model = PPOModel()
        model.actor = actor
        model.actor_old = actor_old
        model.loss = nn.MSELoss()
        return model

    def make_model_3d(self):
        """
        Method for generating a mode for 3d stacked frames
        @return
            the generated model
        """
        actor = PPONetwork3D(self.model_container, self.obs)
        actor_old = PPONetwork3D(self.model_container, self.obs)
        actor_old.load_state_dict(actor.state_dict())
        model = PPOModel()
        model.actor = actor
        model.actor_old = actor_old
        model.loss = nn.MSELoss()
        return model

    def get_state_single_frame(self, state, images):
        """
        Transforms the output of the environment into a input state of the model for single frame type
        @params:
            state   the environment output
            images  the previous images, not used in single frame type
        @return:
            the input state for the model
            the updated images, None in single frame type
        """
        state = self.transformer(state)
        state = state.to(self.device)
        return state, None

    def get_state_2d_stacked_frame(self, next_img, images: ImageContainer):
        """
        Transforms the output of the environment into a input state of the model for 2d stacked frames
        @params:
                state   the environment output
                images  the previous images
        @return:
            the input state for the model
            the updated images
        """
        images.img_3 = images.img_2
        images.img_2 = images.img_1
        state = self.transformer(next_img)
        images.img_1 = state.to(self.device)
        state = torch.cat([images.img_1, images.img_2, images.img_3])
        return state, images

    def get_state_3d_stacked_frame(self, next_img, images: ImageContainer):
        """
        Transforms the output of the environment into a input state of the model for 3d stacked frames
        @params:
            state   the environment output
            images  the previous images
        @return:
            the input state for the model
            the updated images
        """
        images.img_3 = images.img_2
        images.img_2 = images.img_1
        state = self.transformer(next_img)
        images.img_1 = state.to(self.device)
        state = torch.cat([images.img_1, images.img_2, images.img_3])
        return state, images

    def reset_env_single_frame(self):
        """
        Resets the environment for the specific frame type
        Single frame type
        @return:
            the next input state for the mode
            the updated images, none in case of single frame type
        """
        state = self.transformer(self.env.reset()).to(self.device)
        return state, None

    def reset_env_2d_stacked_frame(self):
        """
        Resets the environment for the specific frame type
        2d stacked frames
        @return:
            the next input state for the mode
            the updated images
        """
        images = ImageContainer()
        img_1 = self.transformer(self.env.reset())
        images.img_1 = img_1.to(self.device)
        images.img_2 = torch.zeros(40 * 40 * 3).view(3, 40, 40).to(self.device)
        images.img_3 = torch.zeros(40 * 40 * 3).view(3, 40, 40).to(self.device)
        state = torch.cat([images.img_1, images.img_2, images.img_3])
        return state, images

    def reset_env_3d_stacked_frame(self):
        """
        Resets the environment for the specific frame type
        3d stacked frames
        @return:
            the next input state for the mode
            the updated images
        """
        images = ImageContainer()
        img_1 = self.transformer(self.env.reset())
        images.img_1 = img_1.to(self.device)
        images.img_2 = torch.zeros(40 * 40 * 3).view(3, 40, 40).to(self.device)
        images.img_3 = torch.zeros(40 * 40 * 3).view(3, 40, 40).to(self.device)
        state = torch.stack([images.img_1, images.img_2, images.img_3])
        return state, images

    def save_model(self, pbm_conatiner: PopulationModelContainer):
        """
        Saving a model to a file
        @params:
            pbm_container   the population model container with the model to save
        """
        file = pbm_conatiner.path + str(pbm_conatiner.id) + ".pth"
        saves = ModelSavingContainer()
        saves.model_state = pbm_conatiner.model.actor.state_dict()
        saves.optimizer_state = pbm_conatiner.optimizer.state_dict()
        torch.save(saves, file)

    def load_model(self, path, model, optimizer):
        """
        Loading a model from a file
        @params
            path        the file of the model
            model       the model to overwrite
            optimizer   the optimizer to overwrite
        @returns
            the loaded model
            the loaded optimizer
        """
        file = path
        load = torch.load(file)
        model.load_state_dict(load.model_state)
        optimizer.load_state_dict(load.optimizer_sate)
        return model, optimizer

    def update_statistics(self, info, iteration_statistics, statistics):
        """
        Updates the given statistics with the new info
        @params:
            info    the new information
            iteration_statistics, statistics    the statistic container to update
        """
        if info.success:
            statistics.success += 1
            iteration_statistics.success += 1
            statistics.avg_steps.update(info.num_steps)
            iteration_statistics.avg_steps.update(info.num_steps)
        else:
            statistics.avg_steps.update(self.env.max_steps)
            iteration_statistics.avg_steps.update(self.env.max_steps)
        statistics.count += 1
        iteration_statistics.count += 1
        statistics.avg_reward.update(info.reward)
        iteration_statistics.avg_reward.update(info.reward)
        statistics.avg_reward_penalty.update(info.reward_penalty)
        iteration_statistics.avg_reward_penalty.update(info.reward_penalty)

    def log_statistics(self, iteration, pbm_container: PopulationModelContainer):
        """
        Logging the statistics to tensorboard
        @params:
            iteration   the current iteration
            pbm_container   the container with statistics and writer
        """
        accuracy = pbm_container.statistics.success / pbm_container.statistics.count
        pbm_container.writer.add_scalar("mean accuracy", accuracy, iteration)
        pbm_container.writer.add_scalar(
            "mean reward", pbm_container.statistics.avg_reward.avg, iteration
        )
        pbm_container.writer.add_scalar(
            "mean steps", pbm_container.statistics.avg_steps.avg, iteration
        )
        pbm_container.writer.add_scalar(
            "mean reward penalty",
            pbm_container.statistics.avg_reward_penalty.avg,
            iteration,
        )
        iteration_accuracy = (
                pbm_container.iteration_statistics.success
                / pbm_container.iteration_statistics.count
        )
        pbm_container.writer.add_scalar(
            "mean iteration accuracy", iteration_accuracy, iteration
        )
        pbm_container.writer.add_scalar(
            "mean iteration reward",
            pbm_container.iteration_statistics.avg_reward.avg,
            iteration,
        )
        pbm_container.writer.add_scalar(
            "mean iteration steps",
            pbm_container.iteration_statistics.avg_steps.avg,
            iteration,
        )
        pbm_container.writer.add_scalar(
            "mean iteration reward penalty",
            pbm_container.iteration_statistics.avg_reward_penalty.avg,
            iteration,
        )
        pbm_container.writer.add_scalar(
            "learning rate",
            pbm_container.hyper_container[-1].learning_rate_actor(),
            iteration,
        )
        pbm_container.writer.add_scalar(
            "discount", pbm_container.hyper_container[-1].discount(), iteration
        )
        pbm_container.writer.add_scalar(
            "weight decay", pbm_container.hyper_container[-1].weight_decay(), iteration
        )
        pbm_container.writer.add_scalar(
            "momentum", pbm_container.hyper_container[-1].momentum_sgd(), iteration
        )

        pbm_container.writer.add_scalar("parent", pbm_container.parent[-1], iteration)
        pbm_container.writer.add_scalar("score", pbm_container.score[-1], iteration)
        pbm_container.writer.add_scalar(
            "score history", pbm_container.score_history, iteration
        )
        self.make_grid(iteration, pbm_container)

    def reset_iteration_statistics(self, pbm_container: PopulationModelContainer):
        """
        Resets the statistics for the current iteration
        @params:
            pbm_container   the container of the agent to reset
        """
        pbm_container.iteration_statistics.count = 0
        pbm_container.iteration_statistics.success = 0
        pbm_container.iteration_statistics.avg_reward.reset()
        pbm_container.iteration_statistics.avg_steps.reset()
        pbm_container.iteration_statistics.avg_reward_penalty.reset()

    def make_grid(self, iteration, pbm_container: PopulationModelContainer):
        """
        Let the agent play a full episode in the environment and generates an image grid of all the steps
        @params
            iteration   the current iteration
            pbm_container   agent container
        """
        done = False
        images = []
        state, imgs = self.reset_env()

        img = self.env.render()
        img = self.transformer(img)
        images.append(img)

        while not done:
            state, reward, done, info, memory, imgs = self.make_step(
                pbm_container.model, state, imgs, False
            )
            img = self.env.render()
            img = self.transformer(img)
            images.append(img)

        img_grid = torchvision.utils.make_grid(images)
        pbm_container.writer.add_image("Update", img_grid, global_step=iteration)

    def hyper_string(self, pbm_container: PopulationModelContainer):
        """
        Generates a formatted string with all the hyperparameters of an agent
        @params
            pbm_container   the agent container
        @returns
            the generated string
        """
        return [
            "Hyperparameter:",
            "Environment ID: "
            + self.utility_container.environment_id
            + "; Training Type: "
            + self.algorithm_name
            + "; Frame Type: "
            + self.frameType_name
            + "; Device: "
            + str(self.device)
            + "; Optimizer:"
            + pbm_container.hyper_container[-1].optimizer_name
            + "; Actor Learning Rate: "
            + str(pbm_container.hyper_container[-1].learning_rate_actor)
            + "; Critic Learning Rate: "
            + str(pbm_container.hyper_container[-1].learning_rate_critic)
            + "; Discount: "
            + str(pbm_container.hyper_container[-1].discount)
            + "; Num Epochs: "
            + str(pbm_container.hyper_container[-1].num_epochs)
            + "; EPS Clip: "
            + str(pbm_container.hyper_container[-1].eps_clip)
            + "; Training Steps: "
            + str(pbm_container.hyper_container[-1].environment_steps)
            + "; Episode Update: "
            + str(pbm_container.hyper_container[-1].episode_update)
            + "; Weight decay: "
            + str(pbm_container.hyper_container[-1].weight_decay())
            + "; Momentum: "
            + str(pbm_container.hyper_container[-1].momentum_sgd()),
        ]

    def print_hyper(self, pbm_container: PopulationModelContainer, episode=0):
        """
        Prints a string with all the hyperparameters in the log and the tensorboard
        @params
            pbm_container   the agent container
            episode the current episode, 0 default
        """
        hypers = self.hyper_string(pbm_container)
        pbm_container.writer.add_text(hypers[0], hypers[1], global_step=episode)
        if self.utility_container.logging:
            logging.info(f"Model ID: {pbm_container.id}: {hypers[0]} {hypers[1]}")

    def fit_optimizer(self, pbm_container: PopulationModelContainer):
        """
        Fits the optimizer with the model and the hyperparameters
        @params
            pbm_container   the agent container
        @returns
            the fitted optimizer
        """
        if type(pbm_container.hyper_container[-1].optimizer).__name__ == "SGD":
            optimizer = pbm_container.hyper_container[-1].optimizer(
                [
                    {
                        "params": pbm_container.model.actor.actor.parameters(),
                        "lr": pbm_container.hyper_container[-1].learning_rate_actor(),
                    },
                    {
                        "params": pbm_container.model.actor.critic.parameters(),
                        "lr": pbm_container.hyper_container[-1].learning_rate_critic(),
                    },
                ],
                momentum=pbm_container.hyper_container[-1].momentum(),
                weight_decay=pbm_container.hyper_container[-1].weight_decay(),
            )
        else:
            optimizer = pbm_container.hyper_container[-1].optimizer(
                [
                    {
                        "params": pbm_container.model.actor.actor.parameters(),
                        "lr": pbm_container.hyper_container[-1].learning_rate_actor(),
                    },
                    {
                        "params": pbm_container.model.actor.critic.parameters(),
                        "lr": pbm_container.hyper_container[-1].learning_rate_critic(),
                    },
                ],
                weight_decay=pbm_container.hyper_container[-1].weight_decay(),
            )

        return optimizer

    def make_model_container(self, model_id: int, writer=True):
        """
        Generates a new agent
        @params
            model_id    the id of the agent
        @returns
            the generated agent
        """

        parameter = ProximalPolicyOptimizationContainer()

        trainer_container = PopulationModelContainer()
        trainer_container.id = model_id
        trainer_container.model = self.make_model()
        trainer_container.hyper_container.append(parameter)
        trainer_container.path = self.path + "/" + str(model_id)
        if writer:
            trainer_container.writer = SummaryWriter(trainer_container.path)
            trainer_container.writer.add_graph(
                model=trainer_container.model.actor, input_to_model=self.obs
            )
        trainer_container.optimizer = self.fit_optimizer(trainer_container)
        trainer_container.parent = [model_id]
        return trainer_container

    def make_population(self, writer=True):
        """
        Generates a population of agents
        @params:
            mutate  True if the hyperparameters should be mutated, false if not
        """
        population = []
        for i in range(self.pbt_container.population_size):
            container = self.make_model_container(i, writer)
            container.hyper_container[0] = self.sample_parameter(
                container.hyper_container[0], self.seed + i
            )
            population.append(container)
        return population

    def sample_parameter(
            self, parameter_container: ProximalPolicyOptimizationContainer, seed=0
    ):
        """
        Samples the hyperparameters
        @params
            parameter_container the container containing the hyperparameters
            seed    the random seed
            mutate  True if the learning rate should be mutate
                        False if the learning rate should be sampled
        @returns:
            the hyperparmameter container with the new values
        """
        parameter_container.optimizer = optim.SGD
        parameter_container.optimizer_name = parameter_container.optimizer.__name__

        parameter_container.learning_rate_actor.set(
            parameter_container.learning_rate_actor.sample(seed=seed)
        )
        parameter_container.learning_rate_critic.set(
            parameter_container.learning_rate_critic.sample(seed=seed)
        )

        parameter_container.discount.mutate(self.pbt_container.mutation_cut)
        parameter_container.weight_decay.mutate(self.pbt_container.mutation_cut)
        parameter_container.momentum_sgd.mutate(self.pbt_container.mutation_cut)

        return parameter_container

    def mutate_hyper(self, population, step):
        """
        Mutates the hyperparameters for a whole population
        @params:
            population  the population to mutate
            step        the current step
        """
        for i in range(len(population)):
            if i >= (len(population) - len(population) * self.pbt_container.cut):
                population[i].score_history = 0
                population[i].hyper_container.append(
                    copy.copy(population[0].hyper_container[-1])
                )
                population[i].model.actor.actor.load_state_dict(
                    population[0].model.actor.actor.state_dict()
                )
                population[i].model.actor.critic.load_state_dict(
                    population[0].model.actor.critic.state_dict()
                )
                population[i].model.actor_old.actor.load_state_dict(
                    population[0].model.actor_old.actor.state_dict()
                )
                population[i].model.actor_old.critic.load_state_dict(
                    population[0].model.actor_old.critic.state_dict()
                )
                population[i].parent.append(population[0].id)
                if self.utility_container.logging:
                    logging.info(
                        f"Model {population[i].id} extends model {population[0].id}"
                    )

            else:
                population[i].score_history += self.pbt_container.history_score_cut
                population[i].hyper_container.append(
                    copy.copy(population[i].hyper_container[-1])
                )
                population[i].hyper_container.append(
                    self.mutator(step, population[i].hyper_container[-1])
                )

            self.fit_optimizer(population[i])
            self.log_statistics(step, population[i])
            self.reset_iteration_statistics(population[i])
            self.print_hyper(population[i], step)

    def mutator(self, step, params: ProximalPolicyOptimizationContainer):
        """
        Mutates hyperparameters in a container
        @params:
            step    the current step (currently not used)
            params  a hyperparameter container
        @returns:
            a hyperparameter container with mutated hyperparameters
        """
        hyper = copy.copy(params)

        hyper.learning_rate_actor.mutate(self.pbt_container.mutation_cut)
        hyper.learning_rate_critic.mutate(self.pbt_container.mutation_cut)
        hyper.discount.mutate(self.pbt_container.mutation_cut)
        hyper.weight_decay.mutate(self.pbt_container.mutation_cut)
        hyper.momentum_sgd.mutate(self.pbt_container.mutation_cut)

        return hyper

    def evaluate_all(self, population):
        """
        Evaluates a whole population of agents
        @params:
            population  the population of agents
        """
        for t in tqdm(population):
            self.evaluate_trainer(t, 10)
        self.calc_values(population)
        population.sort(key=lambda tup: tup.score)

    def calc_values(self, population):
        """
        Calculates the rating of a whole population and resets the evaluation
        @params:
            population  a population of agents
        """
        for a in population:
            acc = 1 / ((a.evaluation.success / a.evaluation.count) + 0.000001)
            step_value = a.evaluation.avg_steps.avg / 200
            reward_value = 1 - a.evaluation.avg_reward.avg
            summed_values = (
                    acc + step_value + reward_value + a.evaluation.avg_reward_penalty.avg
            )
            new_score = summed_values / (4 + a.score_history * (1 - a.score[-1]))
            a.score.append(new_score)
            a.evaluation = StatisticContainer()

    def eval_episodes(self, episodes):
        """
        Calculates the number of steps to play until the next evaluation
        @params:
            episodes    the current number of episodes
        @returns
            the number of steps to play until the next evaluation
        """
        return episodes + 10_000

    def evaluate_trainer(self, pbm_container: PopulationModelContainer, evaluates=100):
        """
        Evaluates a trainer by playing several episodes
        @params:
            pbm_container   agent to evaluate
            evaluates   the number of episodes to play for evaluation
        """
        for episode in range(evaluates):
            state, images = self.reset_env()
            done = False
            while not done:
                state, reward, done, info, memory, images = self.make_step(
                    pbm_container.model, state, images, False
                )

                if done:
                    self.update_statistics(
                        info, pbm_container.evaluation, pbm_container.statistics
                    )

        pbm_container.iteration_statistics = copy.copy(pbm_container.evaluation)

    def make_step(self, model, state, images, save_memory=True, memory=None):
        """
        Makes a single step in the environment
        @params:
            model       the model to get get the action
            state       the current input state
            images      previous three images (used for stacked frames)
            save_memory True if the transition should be stored,
                        False otherwise
            memory      the memory to store the transition
        @returns:
            the next input state
            the received reward
            True if the state was terminal, False otherwise
            information on the environment
            the updated memory
            the updated images
        """
        action, memory = self.get_action(state, model, memory, save_memory)
        next_img, reward, done, info = self.env.step(action)
        next_state, images = self.get_state(next_img, images)
        if save_memory:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
        return next_state, reward, done, info, memory, images

    def get_action(self, state, model: PPOModel, memory, save=True):
        """
        Determines the next action and saves the transition
        @params:
            state   the current input state
            model   the model to determine the action
            memory  the memory to save the action
            save    True if the transition should be saved
                    False otherwise
        @returns:
            the next action
            the updated memory
        """
        with torch.no_grad():
            action, action_logprob = model.actor_old.act(state)
        if save:
            memory.states.append(state)
            memory.actions.append(action)
            memory.log_probs.append(action_logprob)

        return action.item(), memory

    def update(
            self, memory, model, optimizer, parameter: ProximalPolicyOptimizationContainer
    ):
        """
        Calculates the loss and updates the model
        @params:
            memory          memory filled with transitions
            model           the model container to update
            optimizer       the optimizer
            parameter       a hyperparameter container
        """
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
                reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (parameter.discount() * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = (
            torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(self.device)
        )
        old_actions = (
            torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(memory.log_probs, dim=0)).detach().to(self.device)
        )

        for _ in range(parameter.num_epochs()):
            log_probs, state_values, dist_entropy = model.actor.evaluate(
                old_states, old_actions
            )

            state_values = torch.squeeze(state_values)

            ratios = torch.exp(log_probs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                    torch.clamp(ratios, 1 - parameter.eps_clip(), 1 + parameter.eps_clip())
                    * advantages
            )

            loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * model.loss(state_values, rewards)
                    - 0.01 * dist_entropy
            )

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

        model.actor_old.load_state_dict(model.actor.state_dict())

    def init_single(self):
        """
        Initializes a baseline run
        @returns:
            a trained agent
        """
        pbm = self.make_model_container(0)
        pbm.hyper_container[0].weight_decay.set(0.000001)
        pbm.optimizer = self.fit_optimizer(pbm)
        self.print_hyper(pbm)
        self.train_single(pbm)
        return pbm

    def train_single(self, pbm: PopulationModelContainer):
        """
        Trains a single agent
        @params:
            pbm     the agent to train
        @returns:
            the trained agent
        """
        parameter = pbm.hyper_container[0]
        model = pbm.model
        optimizer = pbm.optimizer

        state, images = self.reset_env()
        memory = self.make_memory()

        for episode in tqdm(range(parameter.environment_steps())):
            state, reward, done, info, memory, images = self.make_step(
                model, state, images, True, memory
            )
            if done:
                self.update_statistics(info, pbm.iteration_statistics, pbm.statistics)
                state, images = self.reset_env()
                done = False

            if (
                    not episode % parameter.episode_update()
                    and episode >= parameter.episode_update()
            ):
                self.update(memory, model, optimizer, parameter)
                memory.clear()
            if (
                    not episode % self.utility_container.log_iterations()
                    and episode >= self.utility_container.log_iterations()
            ):
                self.log_statistics(
                    episode / self.utility_container.log_iterations(), pbm
                )
                self.reset_iteration_statistics(pbm)
        return pbm

    def init_population_training(self):
        """
        Initializes a population based training
        @returns:
            a trained population
        """
        population = self.make_population()

        for i in population:
            i.hyper_container[0] = self.sample_parameter(
                i.hyper_container[0], seed=self.seed + i.id
            )
        population = self.train_population(population)
        return population

    def train_population(self, population):
        """
        Trains a whole population
        @params:
            population  the population to train
        @returns:
            the trained population
        """
        parameter = population[0].hyper_container[0]
        model = population[0].model
        state, images = self.reset_env()
        memory = self.make_memory()
        next_eval = self.eval_episodes(0)
        for episode in tqdm(range(parameter.environment_steps())):

            state, reward, done, info, memory, images = self.make_step(
                model, state, images, True, memory
            )
            if done:
                done = False
                state, images = self.reset_env()

            if (
                    not episode % parameter.episode_update()
                    and episode >= parameter.episode_update()
            ):
                for agent in population:
                    self.update(
                        memory, agent.model, agent.optimizer, agent.hyper_container[-1]
                    )
                memory.clear()

            if episode == next_eval:
                print(f"episode: {episode}, next eval: {next_eval}")
                self.evaluate_all(population)
                self.mutate_hyper(population, episode)

                next_eval = self.eval_episodes(episode)

                parameter = population[0].hyper_container[0]
                model = population[0].model

        return population

    def transfer_trainer(self):
        """
        Trains a whole population and transfers the best agent to another environment
        """
        population = self.init_population_training()
        self.evaluate_all(population)
        self.env = Gridworld.make("hardcore-10x10-random")
        best = population[0]
        best.writer = SummaryWriter(best.path + "-transfer")
        best.writer.add_graph(model=best.model.actor, input_to_model=self.obs)
        self.train_single(best)

    def population_trainer_transfer(self):
        """
        Trains a population and transfers the whole population to another environment
        """
        self.transfer = False
        population = self.init_population_training()
        self.evaluate_all(population)
        self.env = Gridworld.make("hardcore-10x10-random")
        transfer = self.make_population()
        self.transfer = True
        for i in range(self.pbt_container.population_size):
            transfer[i].model.actor.actor.load_state_dict(
                population[i].model.actor.actor.state_dict()
            )
            transfer[i].model.actor.critic.load_state_dict(
                population[i].model.actor.critic.state_dict()
            )
            transfer[i].model.actor_old.actor.load_state_dict(
                population[i].model.actor_old.actor.state_dict()
            )
            transfer[i].model.actor_old.critic.load_state_dict(
                population[i].model.actor_old.critic.state_dict()
            )

            transfer[i].hyper_container.append(
                copy.copy(population[i].hyper_container[-1])
            )
            transfer[i].writer = SummaryWriter(
                population[i].path + "-transfer" + str(i)
            )
            transfer[i].writer.add_graph(
                model=transfer[i].model.actor, input_to_model=self.obs
            )
            transfer[i].optimizer = self.fit_optimizer(transfer[i])
            transfer[i].parent = [i]
        self.train_population(transfer)

    def init_single_transfer_repeat(self, agents, comment=None):
        """
        Generates multiple baseline agents
        @params:
            agents  the number of agents to train
            comment a logging comment
        """
        if comment is not None:
            logging.info(comment)
        logging.info("Single transfer training")
        for i in range(agents):
            self.env = Gridworld.make("empty-10x10-random")
            pbm = self.make_model_container(i, writer=False)
            pbm.hyper_container[0].weight_decay.set(0.000001)
            pbm.optimizer = self.fit_optimizer(pbm)
            self.print_hyper(pbm)
            pbm = self.train_single(pbm)
            self.env = Gridworld.make("hardcore-10x10-random")
            pbm.writer = SummaryWriter(pbm.path + "-transfer")
            pbm.writer.add_graph(model=pbm.model.actor, input_to_model=self.obs)
            pbm.evaluation = StatisticContainer()
            pbm.statistics = StatisticContainer()
            pbm.score = [0.0]
            pbm.score_history = 0
            self.train_single(pbm)


"""
model_container = ModelContainer()
util_container = UtilityContainer()
util_container.logging = True
util_container.init_logging = True
util_container.environment_id = "empty-10x10-random"
util_container.save = False
util_container.loading = False
pbt = PopulationContainer()
pbt.population_size = 10

trainer = PopulationBasedTrainerPPO(util_container, model_container, pbt)
# trainer.init_single()
# trainer.init_population_training()
# trainer.population_trainer_transfer()
# trainer.init_single_transfer_repeat(20, "Single_transfer")
"""
