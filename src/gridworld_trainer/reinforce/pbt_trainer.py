# @title:    pbt_trainer.py
# @author:   Jan Frederik Liebig
# @date:     02.09.2021

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
    ReinforceContainer,
    FrameType,
    PopulationContainer,
    PopulationModelContainer,
    ImageContainer,
    ModelSavingContainer,
)
from tqdm import tqdm
import torchvision.transforms as transforms
from src.gridworld.gridworld import Gridworld
import logging
import torchvision
from torch.distributions.categorical import Categorical
from src.gridworld_trainer.reinforce.memory import MemoryReinforce, MemoryReinforce3D
from src.gridworld_trainer.reinforce.model import ReinforceNetwork, ReinforceNetwork3D
import os
import copy
import torch.optim as optim

############################################################
# Code


class PopulationBasedTrainerReinforce:
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
        self.algorithm_name = "Reinforce"
        self.frameType_name = ""

        self.get_state = None
        self.reset_env = None
        self.make_model = None
        self.make_memory = None

        self.start_time = int(time.time())

        if self.utility_container.path is None:
            self.path = "../../../logs/gridworld/reinforce/logs-" + str(self.start_time)
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
            self.make_memory = self.make_memory_2d

        if self.utility_container.frame_type == FrameType.Stacked2D:
            self.frameType_name = "2D_stacked_frames"
            self.get_state = self.get_state_2d_stacked_frame
            self.reset_env = self.reset_env_2d_stacked_frame
            self.model_container.num_channel = 9
            self.model_container.num_images = 1
            self.make_model = self.make_model_2d
            self.make_memory = self.make_memory_2d

        if self.utility_container.frame_type == FrameType.Stacked3D:
            self.frameType_name = "3D_stacked_frames"
            self.get_state = self.get_state_3d_stacked_frame
            self.reset_env = self.reset_env_3d_stacked_frame
            self.model_container.num_channel = 3
            self.model_container.num_images = 3
            self.make_model = self.make_model
            self.make_memory = self.make_memory_3d

    def make_memory_2d(self, max_memory_size):
        """
        Generates a memory for the reinforce algorithm for 2d stacked frames and single image frame type
        @param:
            max_memory_size the maximum size for the memory
        @return:
            the generated memory
        """
        memory = MemoryReinforce(
            max_memory_size,
            self.obs_size,
            self.obs_size,
            self.model_container.num_channel,
            self.device,
        )
        return memory

    def make_memory_3d(self, max_memory_size):
        """
        Generates a memory for the reinforce algorithm for 3d stacked frames
        @param:
            max_memory_size the maximum size for the memory
        @return:
            the generated memory
        """
        memory = MemoryReinforce3D(
            max_memory_size,
            self.obs_size,
            self.obs_size,
            self.model_container.num_images,
            self.model_container.num_channel,
            self.device,
        )
        return memory

    def make_model_2d(self):
        """
        Method for generating a mode for 2d stacked frames or a single frame
        @return
            the generated model
        """
        actor = ReinforceNetwork(self.model_container, self.obs)
        return actor

    def make_model_3d(self):
        """
        Method for generating a mode for 3d stacked frames
        @return
            the generated model
        """
        actor = ReinforceNetwork3D(self.model_container, self.obs)
        return actor

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
            images  the previous
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
        saves.model_state = pbm_conatiner.model.state_dict()
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
            pbm_container.hyper_container[-1].learning_rate(),
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

        pbm_container.writer.add_scalar(
            "policy epochs",
            pbm_container.hyper_container[-1].policy_epochs(),
            iteration,
        )
        pbm_container.writer.add_scalar(
            "entropy coefficient",
            pbm_container.hyper_container[-1].entropy_coefficient(),
            iteration,
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
            with torch.no_grad():
                action = Categorical(logits=pbm_container.model.forward(state))
            action = action.sample()
            next_img, reward, done, info = self.env.step(action)
            state, imgs = self.get_state(next_img, imgs)
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
            + "; Policy epochs: "
            + str(pbm_container.hyper_container[-1].policy_epochs())
            + "; Number of updates: "
            + str(pbm_container.hyper_container[-1].num_updates())
            + "; Max memory size: "
            + str(pbm_container.hyper_container[-1].max_memory_size())
            + "; Entropy coefficient: "
            + str(pbm_container.hyper_container[-1].entropy_coefficient())
            + "; Discount: "
            + str(pbm_container.hyper_container[-1].discount())
            + "; Learning rate: "
            + str(pbm_container.hyper_container[-1].learning_rate())
            + "; Batch size: "
            + str(pbm_container.hyper_container[-1].batch_size())
            + "; Optimizer: "
            + pbm_container.hyper_container[-1].optimizer_name
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
                pbm_container.model.parameters(),
                lr=pbm_container.hyper_container[-1].learning_rate(),
                weight_decay=pbm_container.hyper_container[-1].weight_decay(),
                momentum=pbm_container.hyper_container[-1].momentum(),
            )
        else:
            optimizer = pbm_container.hyper_container[-1].optimizer(
                pbm_container.model.parameters(),
                lr=pbm_container.hyper_container[-1].learning_rate(),
                weight_decay=pbm_container.hyper_container[-1].weight_decay(),
            )
        return optimizer

    def make_model_container(self, model_id: int):
        """
        Generates a new agent
        @params
            model_id    the id of the agent
        @returns
            the generated agent
        """

        parameter = ReinforceContainer()

        trainer_container = PopulationModelContainer()
        trainer_container.id = model_id
        trainer_container.model = self.make_model()
        trainer_container.hyper_container.append(parameter)
        trainer_container.path = self.path + "/" + str(model_id)
        trainer_container.writer = SummaryWriter(trainer_container.path)
        trainer_container.writer.add_graph(
            model=trainer_container.model, input_to_model=self.obs
        )
        trainer_container.optimizer = self.fit_optimizer(trainer_container)
        trainer_container.parent = [model_id]
        return trainer_container

    def make_population(self, opti=False):
        """
        Generates a population of agents
        @params:
            mutate  True if the hyperparameters should be mutated, false if not
        """
        population = []
        for i in range(self.pbt_container.population_size):
            container = self.make_model_container(i)
            container.hyper_container[0] = self.sample_parameter(
                container.hyper_container[0], self.seed + i, opti=opti
            )
            population.append(container)
        return population

    def sample_parameter(
        self, parameter_container: ReinforceContainer, seed=0, opti=False
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
        if opti:
            parameter_container.learning_rate.mutate(self.pbt_container.mutation_cut)
        else:
            parameter_container.learning_rate.set(
                parameter_container.learning_rate.sample(seed=seed)
            )

        parameter_container.policy_epochs.mutate(self.pbt_container.mutation_cut)
        parameter_container.entropy_coefficient.mutate(self.pbt_container.mutation_cut)

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
        for i in reversed(range(len(population))):
            if i >= (len(population) - len(population) * self.pbt_container.cut):
                population[i].score_history = 0
                population[i].hyper_container.append(
                    copy.copy(population[0].hyper_container[-1])
                )
                population[i].model.load_state_dict(population[0].model.state_dict())
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

    def mutator(self, step, params: ReinforceContainer):
        """
        Mutates hyperparameters in a container
        @params:
            step    the current step (currently not used)
            params  a hyperparameter container
        @returns:
            a hyperparameter container with mutated hyperparameters
        """
        hyper = copy.copy(params)

        hyper.policy_epochs.mutate(self.pbt_container.mutation_cut)
        hyper.entropy_coefficient.mutate(self.pbt_container.mutation_cut)

        hyper.learning_rate.mutate(self.pbt_container.mutation_cut)
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
        population.sort(key=lambda tup: tup.score[-1])

    def calc_values(self, population):
        """
        Calculates the rating of a whole population and resets the evaluation
        @params:
            population  a population of agents
        """
        for a in population:
            acc = 1 / ((a.evaluation.success / a.evaluation.count) + 0.000001)
            step_value = a.evaluation.avg_steps.avg / 201
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
            episodes    the current number of episodes (currently unused)
        @returns
            the number of steps to play until the next evaluation
        """
        return 5

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
                with torch.no_grad():
                    action = Categorical(logits=pbm_container.model.forward(state))
                action = action.sample()
                next_img, reward, done, info = self.env.step(action)
                state, images = self.get_state(next_img, images)

                if done:
                    self.update_statistics(
                        info, pbm_container.evaluation, pbm_container.statistics
                    )

        pbm_container.iteration_statistics = copy.copy(pbm_container.evaluation)

    def generate_memory(
        self,
        model,
        parameter: ReinforceContainer,
        do_stats=False,
        iteration_statistics=None,
        statistics=None,
    ):
        """
        Plays the game and stores the transitions
        @params:
            model   the model to get the action
            parameter   the parameter container
            do_stats    True if the statistics should be logged
                        False otherwise
            iteration_statistics    statistics for one iteration
            statistics  all time statistics
        @returns:
            the memory with generated transitions
        """
        memory = self.make_memory(parameter.max_memory_size())
        done = False
        info = None
        state, images = self.reset_env()
        for episode in tqdm(range(parameter.max_memory_size())):
            if done:
                next_state, images = self.reset_env()
                if do_stats:
                    self.update_statistics(info, iteration_statistics, statistics)
            else:
                next_state = state

            action, log_prob = self.get_action(next_state, model)
            next_img, reward, done, info = self.env.step(action)
            next_state, images = self.get_state(next_img, images)

            memory.insert(
                episode,
                torch.tensor(done, dtype=torch.float32),
                action,
                log_prob,
                torch.tensor(reward, dtype=torch.float32),
                state,
            )

            state = next_state

        return memory

    def get_action(self, state, model):
        """
        Determines the next action
        @params:
            state   the current input state
            model   the model to determine the action
        @returns:
            the next action
            the logarithmic probability
        """
        with torch.no_grad():
            logits = model.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def update(self, rollouts, model, optimizer, parameter: ReinforceContainer):
        """
        Calculates the loss and updates the model
        @params:
            rollouts          memory filled with transitions
            model           the model to update
            optimizer       the optimizer
            parameter       a hyperparameter container
        """
        for epoch in tqdm(range(parameter.policy_epochs())):
            data = rollouts.batch_sampler(parameter.batch_size())
            for sample in data:
                actions_batch, returns_batch, obs_batch = sample
                actions_batch = actions_batch.to(self.device)
                log_probs_batch, entropy_batch = self.evaluate_actions(
                    obs_batch, actions_batch, model
                )
                log_probs_batch = log_probs_batch.to(self.device)
                returns_batch = returns_batch.to(self.device)

                policy_loss = -(log_probs_batch * returns_batch).mean()

                entropy_loss = -entropy_batch.mean()
                loss = policy_loss + parameter.entropy_coefficient() * entropy_loss
                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()

    def evaluate_actions(self, state, action, model):
        """
        Evaluates the taken actions
        @params:
            state   the state to evaluate with
            action  the action to evaluate
            model   the model to evaluate with
        @returns
            logarithmic probability of the action
            the entropy of the distribution
        """
        logits = model.forward(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action.squeeze(-1)).view(-1, 1)
        entropy = dist.entropy().view(-1, 1)
        return log_prob, entropy

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

        for updates in range(parameter.num_updates()):
            memory = self.generate_memory(
                model, parameter, True, pbm.iteration_statistics, pbm.statistics
            )
            memory.compute_returns(parameter.discount())
            self.update(memory, model, optimizer, parameter)
            memory.reset()
            self.log_statistics(updates * parameter.max_memory_size(), pbm)
            self.reset_iteration_statistics(pbm)

        return pbm

    def init_population_training(self, opti=False):
        """
        Initializes a population based training
        @returns:
            a trained population
        """
        population = self.make_population(opti=opti)

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
        episodes = 0
        done_training = False
        while not done_training:
            next_episodes = self.eval_episodes(episodes)
            for eps in range(next_episodes):
                memory = self.generate_memory(
                    population[0].model, population[0].hyper_container[-1]
                )
                for t in population:
                    t_memory = copy.copy(memory)
                    t_memory.compute_returns(t.hyper_container[-1].discount())
                    self.update(t_memory, t.model, t.optimizer, t.hyper_container[-1])
                    t_memory.reset()
                memory.reset()

                if episodes == self.pbt_container.evaluation_steps:
                    done_training = True
                episodes += 1

            self.evaluate_all(population)
            self.mutate_hyper(population, episodes)

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
        best.writer.add_graph(model=best.model, input_to_model=self.obs)
        self.train_single(best)

    def population_trainer_transfer(self, best_pop=False, comment=None):
        """
        Trains a population and transfers the whole population to another environment
        @params:
            best_pop    True if the transfer population should be copies of the best agent
                        False if the whole population should be transferred
            comment     comment for logging
        """
        if comment is not None:
            logging.info(comment)
        population = self.init_population_training(opti=True)
        self.evaluate_all(population)
        self.env = Gridworld.make("hardcore-10x10-random")
        logging.info("pb transfer training")
        transfer = []
        if best_pop:
            for i in range(self.pbt_container.population_size):
                transfer.append(copy.copy(population[0]))
                transfer[i].writer = SummaryWriter(
                    transfer[i].path + "-transfer" + str(i)
                )
                transfer[i].writer.add_graph(
                    model=transfer[i].model, input_to_model=self.obs
                )
                transfer[i].optimizer = self.fit_optimizer(transfer[i])
                transfer[i].evaluation = StatisticContainer()
                transfer[i].statistics = StatisticContainer()
                transfer[i].score = [0.0]
                transfer[i].score_histoy = 0
                transfer[i].parent = [i]
        else:
            for i in range(self.pbt_container.population_size):
                transfer.append(copy.copy(population[i]))
                transfer[i].writer = SummaryWriter(
                    transfer[i].path + "-transfer" + str(i)
                )
                transfer[i].writer.add_graph(
                    model=transfer[i].model, input_to_model=self.obs
                )
                transfer[i].optimizer = self.fit_optimizer(transfer[i])
                transfer[i].evaluation = StatisticContainer()
                transfer[i].statistics = StatisticContainer()
                transfer[i].score = [0.0]
                transfer[i].score_histoy = 0
                transfer[i].parent = [i]
        self.train_population(transfer)

    def init_single_transfer(self, comment=None):
        """
        Initializes a single baseline agent
        @param:
            comment a logging comment
        """
        if comment is not None:
            logging.info(comment)
        logging.info("Single transfer training")

        pbm = self.make_model_container(0)
        pbm.hyper_container[0].weight_decay.set(0.000001)
        pbm.optimizer = self.fit_optimizer(pbm)
        self.print_hyper(pbm)
        pbm = self.train_single(pbm)
        self.env = Gridworld.make("hardcore-10x10-random")
        pbm.writer = SummaryWriter(pbm.path + "-transfer")
        pbm.writer.add_graph(model=pbm.model, input_to_model=self.obs)
        pbm.evaluation = StatisticContainer()
        pbm.statistics = StatisticContainer()
        pbm.score = [0.0]
        pbm.score_history = 0
        self.train_single(pbm)

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
            pbm = self.make_model_container(i)
            pbm.hyper_container[0].weight_decay.set(0.000001)
            pbm.optimizer = self.fit_optimizer(pbm)
            self.print_hyper(pbm)
            pbm = self.train_single(pbm)
            self.env = Gridworld.make("hardcore-10x10-random")
            pbm.writer = SummaryWriter(pbm.path + "-transfer")
            pbm.writer.add_graph(model=pbm.model, input_to_model=self.obs)
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
# util_container.environment_id = "hardcore-10x10-random"

util_container.save = False
util_container.loading = False
pbt = PopulationContainer()
pbt.population_size = 10

trainer = PopulationBasedTrainerReinforce(util_container, model_container, pbt)
# trainer.init_single()
# trainer.init_population_training()
# trainer.transfer_trainer()
# trainer.init_single_transfer()

# trainer.population_trainer_transfer(best_pop=False, comment="\ntransfer type: best agent \n")
# trainer.population_trainer_transfer(best_pop=True, comment="\ntransfer type: whole pop \n")
# trainer.init_single_transfer_repeat(10)
"""

