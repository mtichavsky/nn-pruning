import argparse
import copy
import logging
import os
import time
from pathlib import Path
import random
import json

import torch
import torchvision
from deap import creator, base
from deap import tools
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import transforms

from model import load_model, set_mask_on

# Evolution
POPULATION_SIZE = 7
POP_INDIVIDUALS_SLIGHTLY_MUTATED = 3
CXPB = 0.35
# Mutate
FLIP_BITS_WHEN_MUTATE = 18  # p = X / chromosome_len
BATCH_SIZE = int(os.getenv("BATCH_SIZE", default=64))
TUNING_EPOCHS = 16
PATIENCE = int(os.getenv("PATIENCE", default=3))
ACCURACY_THRESHOLD = 0.85
RANDOM_SEARCH_MINIMUM_MODEL_SIZE = 2000

# Training
TRAIN_EPOCHS = int(os.getenv("NOF_EPOCHS", default=30))
SAVE_MODEL_INTERVAL = int(os.getenv("SAVE_MODEL_INTERVAL", default=5))

# General
DATASET_DIR = os.getenv("DATASET_DIR", default="./data")
MODEL_OUT_DIR = Path("artifacts")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
def walltime_to_seconds(walltime_str):
    if walltime_str is None:
        return None
    hours, minutes, seconds = map(int, walltime_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds
WALLTIME = walltime_to_seconds(os.getenv("WALLTIME", default=None))

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s  %(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)
torch.manual_seed(42)
random.seed(42)


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
        root=DATASET_DIR, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATASET_DIR, train=False, download=True, transform=transform_test
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
    train_parser.add_argument("--epochs", type=int, default=int(TRAIN_EPOCHS))
    train_parser.add_argument("--model", type=str, default=None, help="Base model path")
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
        "--model", type=str, default=None, help="Base model path"
    )
    evolve_parser.add_argument(
        "--optimizer", type=str, default=None, help="Optimizer path"
    )
    evolve_parser.add_argument(
        "--scheduler", type=str, default=None, help="Scheduler path"
    )
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument(
        "--model", type=str, default=None, help="Base model path"
    )
    random_parser = subparsers.add_parser("random", help="Perform random search")
    random_parser.add_argument("--individuals", type=int, default=20)
    random_parser.add_argument(
        "--model", type=str, default=None, help="Base model path"
    )
    random_parser.add_argument(
        "--optimizer", type=str, default=None, help="Optimizer path"
    )
    random_parser.add_argument(
        "--scheduler", type=str, default=None, help="Scheduler path"
    )
    return parser.parse_args()

def generate_seed_population(chromosome_len, population_size):
    population = []

    full_model = creator.Individual([1] * chromosome_len)
    population.append(full_model)

    for _ in range(POP_INDIVIDUALS_SLIGHTLY_MUTATED):
        individual = creator.Individual([1] * chromosome_len)
        prune_idx = random.sample(
            range(chromosome_len), k=random.randint(1, chromosome_len // 4)
        )
        for idx in prune_idx:
            individual[idx] = 0
        population.append(individual)

    while len(population) < population_size:
        individual = toolbox.individual()
        population.append(individual)

    return population

def evaluate(model, mask=None):
    model.eval()
    if mask:  # Just in case, this should be fixed in train_episode TODO
        set_mask_on(model, mask)
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


def evaluate_evolution(base_model, optimizer_path, scheduler_path, individual):
    model = copy.deepcopy(base_model)
    set_mask_on(model, individual)
    correct, total = tune(model, TUNING_EPOCHS, optimizer_path, scheduler_path, early_stopping=True, mask=individual)
    logger.info(f"[Evaluated] {sum(individual)}: {correct / total}")
    accuracy = float(correct) / total
    if accuracy < ACCURACY_THRESHOLD: # penalize anything less than 0.85, might increase
        accuracy = 0.1 * accuracy
    return sum(individual), accuracy

def run_evolution(tb, ngens, mutation_prob, start_time):
    pop = tb.population(POPULATION_SIZE)
    fitness = map(tb.evaluate, pop)
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    all_solutions = [copy.deepcopy(ind) for ind in pop]
    best_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

    for gen in range(ngens):
        if WALLTIME and time.time() - start_time > WALLTIME - 50 * 60:
            logger.info(f"Reached walltime -50minutes, stopping evolution")
            break
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        all_solutions.extend(copy.deepcopy(ind) for ind in offspring)
        best_front = tools.sortNondominated(best_front + offspring, k=2*POPULATION_SIZE, first_front_only=True)[0]
        pop[:] = toolbox.select(pop + offspring, POPULATION_SIZE)

        logger.info(f"Generation {gen}:")
        fits = [ind.fitness.values for ind in pop]
        logger.info("Preview individuals from generation (sizes, accuracy):")
        for f in fits[:5]:  # preview
            logger.info(f)
        save_pareto_solutions(best_front, f"{gen}")
        save_pareto_solutions(all_solutions, f"{gen}_all")


def train_episode(model, optimizer, scheduler, criterion, mask=None):
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
        if mask:
            set_mask_on(model, mask)

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
        optimizer.load_state_dict(
            torch.load(optimizer_path, map_location=torch.device(device))
        )
    if scheduler_path:
        logger.info(f"Loading scheduler from {scheduler_path}")
        scheduler.load_state_dict(
            torch.load(scheduler_path, map_location=torch.device(device))
        )
    criterion = CrossEntropyLoss()
    for i in range(epochs):
        logger.info(f"Epoch {i + 1}/{epochs}")
        avg_loss, acc = train_episode(model, optimizer, scheduler, criterion)
        logger.info("Train loss: {:.4f}, acc: {:.4f}".format(acc, avg_loss))
        correct, total = evaluate(model)
        logger.info(f"Test dataset precision: {correct / total}")
        if save and i % SAVE_MODEL_INTERVAL == 0:
            torch.save(model.state_dict(), MODEL_OUT_DIR / f"model.{i:02d}.pth")
            torch.save(optimizer.state_dict(), MODEL_OUT_DIR / f"optimizer.{i:02d}.pth")
            torch.save(scheduler.state_dict(), MODEL_OUT_DIR / f"scheduler.{i:02d}.pth")


def tune(model, epochs, optimizer_path, scheduler_path, early_stopping=False, mask=None):
    optimizer = SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True
    )
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.1, patience=3, threshold=0.001, mode="max"
    )
    if optimizer_path:
        logger.info(f"Loading optimizer from {optimizer_path}")
        optimizer.load_state_dict(
            torch.load(optimizer_path, map_location=torch.device(device))
        )
    if scheduler_path:
        logger.info(f"Loading scheduler from {scheduler_path}")
        scheduler.load_state_dict(
            torch.load(scheduler_path, map_location=torch.device(device))
        )
    criterion = CrossEntropyLoss()

    best_correct = None
    total = None
    patience_counter = 0
    for i in range(epochs):
        avg_loss, acc = train_episode(model, optimizer, scheduler, criterion, mask=mask)
        logger.info("Train accuracy: {:.4f}, loss: {:.4f}".format(acc, avg_loss))
        correct, total = evaluate(model, mask=mask)
        logger.info(f"Test accuracy: {correct / total}")
        if early_stopping and best_correct and correct < best_correct:
            patience_counter += 1
            logger.info(f"Patience_counter++: {patience_counter}")
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {i}")
                return best_correct, total
        else:
            patience_counter = 0
            best_correct = correct
    logger.info(f"Trained for full {epochs} epochs")
    return best_correct, total


def save_pareto_solutions(front, generation=None):
    solutions = []
    for idx, individual in enumerate(front):
        solution = {
            "mask": individual,
            "fitness": individual.fitness.values,
            "size": individual.fitness.values[0],
            "accuracy": individual.fitness.values[1],
        }
        solutions.append(solution)

    filename = (
        MODEL_OUT_DIR
        / f'pareto_solutions_gen_{generation if generation else "final"}.json'
    )
    with open(filename, "w") as f:
        json.dump(solutions, f)


def run_random_search(base_model, optimizer_path, scheduler_path, chromosome_len, individuals, start_time):
    """
    Perform random search with the same evaluation budget as evolution.
    """
    all_solutions = []
    logger.info(f"Starting random search with {individuals} evaluations budget")
    for i in range(individuals):
        if WALLTIME and  time.time() - start_time > WALLTIME - 50 * 60:
            logger.info(f"Reached walltime -50minutes, stopping random search")
            break

        # Generate a random individual
        individual = [0] * chromosome_len
        num_ones = random.randint(RANDOM_SEARCH_MINIMUM_MODEL_SIZE, chromosome_len)
        ones_positions = random.sample(range(chromosome_len), num_ones)
        for pos in ones_positions:
            individual[pos] = 1

        model = copy.deepcopy(base_model)
        set_mask_on(model, individual)
        correct, total = tune(model, TUNING_EPOCHS , optimizer_path, scheduler_path,
                              early_stopping=True, mask=individual)
        accuracy = float(correct) / total

        if accuracy < ACCURACY_THRESHOLD:  # Same penalty as in evolution
            accuracy = 0.1 * accuracy

        solution = {
            "mask": individual,
            "size": sum(individual),
            "accuracy": accuracy,
            "fitness": [sum(individual), accuracy]
        }
        logger.info(f"[Evaluated solution] {solution['size']}: {solution['accuracy']}")
        all_solutions.append(solution)

        # Save intermediate results
        if i % (POPULATION_SIZE * 5) == 0:  # Save every few "generations" equivalent
            with open(MODEL_OUT_DIR / f'random_search_solutions_{i}.json', 'w') as f:
                json.dump(all_solutions, f)

    # Save final results
    with open(MODEL_OUT_DIR / 'random_search_solutions_final.json', 'w') as f:
        json.dump(all_solutions, f)


if __name__ == "__main__":
    logger.info("starting execution")
    start_time = time.time()
    args = parse_args()
    model, chromosome_len = load_model(device, args.model, logger)
    train_loader, test_loader = get_dataset_loaders()

    if args.command == "train":
        train(
            model,
            args.epochs,
            scheduler_path=args.scheduler,
            optimizer_path=args.optimizer,
        )
    elif args.command == "eval":
        logger.info("Starting evaluation")
        mask = [1] * chromosome_len
        # mask = [1] * (int(chromosome_len / 2) - 1000) + [0] * 1000 + int(chromosome_len / 2) * [1]
        masked_model = copy.deepcopy(model)
        set_mask_on(masked_model, mask)
        masked_correct, masked_total = evaluate(model, mask=mask)
        logger.info(f"[Evaluated] {args.model}: {masked_correct / masked_total}")
    elif args.command == "evolve":
        # Normalizing it to the same range, though it should not have any significant effect
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 4800.0))
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
        #
        toolbox.register("evaluate", evaluate_evolution, model, args.optimizer, args.scheduler)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=FLIP_BITS_WHEN_MUTATE / chromosome_len)
        toolbox.register("population", generate_seed_population, chromosome_len)
        toolbox.register("select", tools.selNSGA2)

        run_evolution(toolbox, args.gens, args.mutation_prob, start_time=start_time)
    elif args.command == "random":
        run_random_search(model, args.optimizer, args.scheduler, chromosome_len, args.individuals, start_time=start_time)
    else:
        raise ValueError(f"Unknown command {args.command}")
