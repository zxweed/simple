# -*- coding: utf-8 -*-

import numpy as np
from inspect import getfullargspec
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, Manager
from random import randrange, randint, uniform
from contextlib import closing
from tqdm.auto import tqdm
from itertools import product, zip_longest
from functools import partial
from alquant.utils import tqdmParallel, pmap, common_type
from joblib import delayed
from psutil import cpu_percent

from deap import creator, base, tools
from deap.algorithms import varAnd
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

class Opt:
    def __init__(self, target):
        self.target = target.__wrapped__ if hasattr(target, '__wrapped__') else target
        spec = getfullargspec(self.target)
        self.args = spec.args
        self.log_columns = self.args
        self.annotations = spec.annotations
        defaults = [] if spec.defaults is None else reversed(spec.defaults)
        self.defaults = dict(reversed(list(zip_longest(reversed(self.args), defaults, fillvalue=100))))
        self.log = []


def inclusive_range(*args):
    """
    Generates an inclusive range of numbers based on the given arguments.
    
    Args:
        *args: The arguments can be passed in any of the following formats:
            - (stop): Generates numbers from 0 to stop-1 with a step of 1.
            - (start, stop): Generates numbers from start to stop-1 with a step of 1.
            - (start, stop, step): Generates numbers from start to stop-1 with a step of step.
    
    Yields:
        int: The next number in the inclusive range.
    
    Raises:
        TypeError: If no arguments are passed or if more than 3 arguments are passed.
    """
    nargs = len(args)
    if nargs == 0:
        raise TypeError("you need to write at least a value")
    elif nargs == 1:
        stop = args[0]
        start = 0
        step = 1
    elif nargs == 2:
        (start, stop) = args
        step = 1
    elif nargs == 3:
        (start, stop, step) = args
    else:
        raise TypeError("Inclusive range was expected at most 3 arguments,got {}".format(nargs))
    i = start
    while i <= stop:
        yield i
        i += step


class GridOpt(Opt):
    """Full grid optimization engine"""

    def fullSearch(self):
        X = product(*(inclusive_range(*v) if isinstance(v, tuple) else (v,) for v in self.defaults.values()))
        grid = [dict(zip(self.args, x)) for x in X]

        with tqdmParallel(total=len(grid), backend='multiprocessing') as P:
            FUNC = delayed(self.target)
            log = P(FUNC(**arg) for arg in grid)

        self.log_columns += list(log[0].keys())
        self.log = [(*x.values(), *r.values()) for x, r in zip(grid, log)]
        return max([x[0] for x in self.log])


class GeneOpt(Opt):
    """Genetic optimization engine"""

    def genRandom(self, arg):
        """returns a random value for the argument name, depending on its type and constraints"""
        if arg in self.annotations:
            p = self.defaults[arg]
            if self.annotations[arg] == int:
                return randrange(*p) if type(p) == tuple else randint(0, p+1)
            elif self.annotations[arg] == float:
                return uniform(p[0], p[1]) if type(p) == tuple else uniform(0, p)

        # default arg type = float(0..100)
        return uniform(0, 100)

    def genIndividual(self):
        return [self.genRandom(arg) for arg in self.args]

    def evalOneMax(self, individual):
        result_dict = self.target(*individual)
        return result_dict,

    def maximize(self, population_size=128, generations=5, what: str='Profit', callback: callable = None):
        """
        Maximizes the fitness value of the target function using a genetic algorithm.

        Args:
            population_size (int): The size of the population. Default is 128.
            generations (int): The number of generations to run the algorithm. Default is 5.
            what (str): The value name to maximize (if the fitness function returns dict). Default is 'Profit'.
            callback (callable): A callback function to be called after each generation (can be used to update chart)

        Returns:
            dict: A dictionary containing the best individual found by the algorithm.
        """
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, self.genIndividual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evalOneMax)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=5)
        population = toolbox.population(n=population_size)

        with closing(Pool()) as P:
            toolbox.register("map", P.map)

            pbar = tqdm(range(generations))
            for _ in pbar:
                offspring = varAnd(population, toolbox, cxpb=0.5, mutpb=0.3)
                fits = toolbox.map(toolbox.evaluate, offspring)

                if callback is not None:
                    callback(offspring, fits)

                for fit, ind in zip(fits, offspring):
                    f = fit[0]   # fitness value is tuple always
                    if type(f) == dict:
                        fitness_value = f[what] if what in f else 0
                        fitness_tuple = (fitness_value,)

                        # only ordinal typed items stored in the log list
                        filtered_dict = dict(filter(lambda x: type(x[1]) in [str, int, float, np.float64], f.items()))
                        self.log_columns += list(filter(lambda x: x not in self.log_columns, filtered_dict))
                        self.log.append((*ind, *filtered_dict.values()))

                    else:
                        fitness_value = f
                        fitness_tuple = (fitness_value,)
                        if 'Fitness' not in self.log_columns:
                            self.log_columns += ['Fitness']
                        self.log.append((*ind, f))

                    ind.fitness.values = fitness_tuple

                population = toolbox.select(offspring, k=len(population))
                pbar.set_postfix(dict(fit=f'{fitness_value:,.2f}', cpu=f'{cpu_percent():1.0f}%'))

        del creator.FitnessMax
        del creator.Individual

        return dict(zip(self.args, tools.selBest(population, k=1)[0]))


def _suggest(trial, key, value):
    """create trial.suggest_XXX object for specified value"""

    if type(value) is list:
        return trial.suggest_categorical(key, value)
    elif type(value) is tuple and len(value) >= 2:
        if isinstance(value[0], int) and isinstance(value[1], int):
            return trial.suggest_int(key, value[0], value[1])
        else:
            return trial.suggest_float(key, value[0], value[1])


class TOptuna(Opt):
    """Optuna-based bayesian optimization class"""

    def __init__(self, target, storage_url='sqlite:///optuna.db', study_name=None, what=None, direction='maximize'):
        super().__init__(target)
        self.log = Manager().list()
        self.what = what
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction=direction
        )

    def objective(self, trial):
        """run one trial with optuna-sampled parameter values"""
        params = {key: _suggest(trial, key, value) for key, value in self.defaults.items()}
        value = self.target(**params)
        self.log.append((*params.values(), value))

        if type(value) is dict:
            if self.what is None:
                result = value[list(value.keys())[0]]   # use the first dict value as fitness
            else:
                result = value[self.what]   # use the 'what' field value
        elif type(value) is list or type(value) is tuple:
            if self.what is None:
                result = value[0]
            else:
                result = value[self.what]
        else:
            result = value

        return result

    def bestValue(self):
        return self.study.best_value

    def run(self, attempts=128, n_jobs=-1, n_trials=1, backend='multiprocessing'):
        """parallel optuna-based bayesian optimization"""
        pmap(
            partial(self.study.optimize, n_trials=n_trials),
            [self.objective] * attempts,
            postfix={self.what: self.bestValue} if self.what is not None else {'value': self.bestValue},
            n_jobs=n_jobs,
            backend=backend
        )
        return self.study.best_params


def optrun(target, what=None, direction='maximize', **kwargs):
    """single-line func wrapper for TOptuna"""
    G = TOptuna(target, what=what, direction=direction)
    G.run(**kwargs)

    # expand dicts from the log to values if necessary
    value = G.log[0][-1]
    if type(value) is dict:
        result_list = [(*p[:-1], *p[-1].values()) for p in G.log]
        columns = G.args + list(value.keys())
    else:
        result_list = G.log[:]
        columns = G.args + ['value']

    # create structured array for result
    dtype = [(col, common_type(set([type(r[i]) for r in result_list]))) for i, col in enumerate(columns)]
    return np.array(result_list, dtype=dtype).view(np.recarray)
