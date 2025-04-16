import argparse
import logging
import os
from pathlib import Path
import random
from typing import Optional

import torch
import torchvision
from deap import creator, base
from deap import tools
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import transforms

from model import load_model, set_mask_on

BATCH_SIZE = int(os.getenv("BATCH_SIZE", default=64))
RESNET_EPOCHS = int(os.getenv("NOF_EPOCHS", default=30))
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_OUT_DIR = Path("artifacts")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
# TODO
# POPULATION_SIZE = 50
POPULATION_SIZE = 15
INDIVIDUAL_EPOCHS = 5

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s  %(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)
torch.manual_seed(42)
random.seed(42)


def evaluate(model):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predictions = outputs.max(1)

            total += len(inputs)
            correct += predictions.eq(targets).sum().item()
        return correct, total


def get_dataset_loaders():
    # https://github.com/kuangliu/pytorch-cifar
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    # more workers?
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    return train_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--epochs", type=int, default=int(RESNET_EPOCHS))
    train_parser.add_argument(
        "--model", type=str, default=None, help="Base model path"
    )
    train_parser.add_argument(
        "--optimizer", type=str, default=None, help="Optimizer path"
    )
    train_parser.add_argument(
        "--scheduler", type=str, default=None, help="Scheduler path"
    )

    evolve_parser = subparsers.add_parser("evolve")
    evolve_parser.add_argument("--gens", type=int, default=20)
    evolve_parser.add_argument("--mutation-prob", type=float, default=0.2)
    evolve_parser.add_argument(
        "--model", type=Optional[str], default=None, help="Base model path"
    )

    return parser.parse_args()


def generate_seed_population(chromosome_len, population_size):
    population = []

    full_model = creator.Individual([1] * chromosome_len)
    population.append(full_model)

    for _ in range(5):
        individual = creator.Individual([1] * chromosome_len)
        prune_idx = random.sample(
            range(chromosome_len), k=random.randint(1, chromosome_len // 10)
        )
        for idx in prune_idx:
            individual[idx] = 0
        population.append(individual)

    while len(population) < population_size:
        individual = toolbox.individual()
        population.append(individual)

    return population


def evaluate_evolution(base_model, individual):
    set_mask_on(base_model, individual)
    train(model, INDIVIDUAL_EPOCHS, False)
    correct, total = evaluate(base_model)
    return sum(individual), float(correct) / total


def run_evolution(tb, ngens, mutation_prob):
    pop = tb.population(POPULATION_SIZE)
    fitness = map(tb.evaluate, pop)
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    for gen in range(ngens):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = toolbox.select(pop + offspring, POPULATION_SIZE)

        logger.info(f"Generation {gen}:")
        fits = [ind.fitness.values for ind in pop]
        for f in fits[:5]:  # preview
            logger.info(f)
    return tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]


def train_episode(optimizer, scheduler, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        # predictions = outputs.argmax(dim=1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    acc = 100.0 * correct / total
    avg_loss = train_loss / len(train_loader)
    scheduler.step(acc)  # adjust based on validation accuracy/loss if applicable
    return avg_loss, acc


def train(model, epochs, save=True, optimizer_path=None, scheduler_path=None):
    optimizer = SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True
    )
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.1, patience=3, threshold=0.001, mode="max"
    )
    if optimizer_path:
        logger.info(f"Loading optimizer from {optimizer_path}")
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=torch.device(device)))
    if scheduler_path:
        logger.info(f"Loading scheduler from {scheduler_path}")
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=torch.device(device)))
    criterion = CrossEntropyLoss()
    for i in range(epochs):
        logger.info(f"Epoch {i + 1}/{RESNET_EPOCHS}")
        avg_loss, acc = train_episode(optimizer, scheduler, criterion)
        logger.info("Train loss: {:.4f}, acc: {:.4f}".format(avg_loss, acc))
        correct, total = evaluate(model)
        logger.info(f"Test dataset precision: {correct / total}")
        if save and i % 5 == 0:
            torch.save(model.state_dict(), MODEL_OUT_DIR / f"model.{i:02d}.pth")
            torch.save(optimizer.state_dict(), MODEL_OUT_DIR / f"optimizer.{i:02d}.pth")
            torch.save(scheduler.state_dict(), MODEL_OUT_DIR / f"scheduler.{i:02d}.pth")


if __name__ == "__main__":
    logger.info("starting execution")
    args = parse_args()
    model, chromosome_len = load_model(device, args.model, logger)
    train_loader, test_loader = get_dataset_loaders()

    if args.command == "train":
        train(model, RESNET_EPOCHS, scheduler_path=args.scheduler, optimizer_path=args.optimizer)
    elif args.command == "evolve":
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_bool,
            chromosome_len,
        )
        toolbox.register("population", generate_seed_population, chromosome_len)
        toolbox.register("evaluate", evaluate_evolution, model)
        toolbox.register("mutate", tools.mutFlipBit, indpb=2 / chromosome_len)

        out = run_evolution(toolbox, args.gens, args.mutation_prob)
        logger.info(out)
