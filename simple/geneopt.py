# -*- coding: utf-8 -*-

from deap import creator, base, tools
from deap.algorithms import varAnd, eaSimple
import inspect
from multiprocessing.pool import Pool
from random import randrange, randint, uniform
from contextlib import closing
from tqdm.auto import tqdm
from itertools import zip_longest
import numpy as np
import pandas as pd
from itertools import product
from simple.pretty import tqdmParallel
from joblib import delayed
from psutil import cpu_percent


class Opt:
    def __init__(self, target):
        self.target = target.__wrapped__ if hasattr(target, '__wrapped__') else target
        spec = inspect.getfullargspec(self.target)
        self.args = spec.args
        self.log_columns = self.args
        self.annotations = spec.annotations
        defaults = [] if spec.defaults is None else reversed(spec.defaults)
        self.defaults = dict(reversed(list(zip_longest(reversed(self.args), defaults, fillvalue=100))))
        self.log = []


def inclusive_range(*args):
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
        raise TypeError("Inclusive range was expected at most 3 arguments, got {}".format(nargs))
    i = start
    while i <= stop:
        yield i
        i += step


class GridOpt(Opt):
    """Full grid optimization engine"""

    def fullSearch(self, callback: callable = None):
        # create list with all parameter combinations
        X = product(*(inclusive_range(*v) if type(v) == tuple else v for v in self.defaults.values()))
        grid = [dict(zip(self.args, x)) for x in X]

        with tqdmParallel(total=len(grid), backend='multiprocessing') as P:
            FUNC = delayed(self.target)
            log = P(FUNC(**arg) for arg in grid)

        self.log_columns += list(log[0].keys())
        self.log = [(*x.values(), *r.values()) for x, r in zip(grid, log)]
        return max([x[0] for x in self.log])
    
    @property
    def report(self):
        return pd.DataFrame(self.log, columns=self.log_columns)


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

    def maximize(self, population_size=128, generations=5, callback: callable = None):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register("individual", tools.initIterate, creator.Individual, self.genIndividual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evalOneMax)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        population = toolbox.population(n=population_size)

        with closing(Pool()) as P:
            toolbox.register("map", P.map)

            pbar = tqdm(range(generations))
            for _ in pbar:
                offspring = varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
                fits = toolbox.map(toolbox.evaluate, offspring)

                if callback is not None:
                    callback(offspring, fits)

                for fit, ind in zip(fits, offspring):
                    f = fit[0]   # fitness value is tuple always
                    if type(f) == dict:
                        fitness_value = f['Profit'] if 'Profit' in f else 0
                        fitness_tuple = (fitness_value,)

                        # only ordinal typed items stored in the log list
                        filtered_dict = dict(filter(lambda x: type(x[1]) in [str, int, float, np.float64], f.items()))
                        self.log_columns += list(filter(lambda x: x not in self.log_columns, filtered_dict))
                        self.log.append((*ind, *filtered_dict.values()))

                    else:
                        fitness_tuple = f,
                        self.log.append((*ind, f))

                    ind.fitness.values = fitness_tuple

                population = toolbox.select(offspring, k=len(population))
                pbar.set_postfix(dict(fit=f'{fitness_value:,.2f}', cpu=f'{cpu_percent():1.0f}%'))

        del creator.FitnessMax
        del creator.Individual

        return dict(zip(self.args, tools.selBest(population, k=1)[0]))

    def progress(self, x):
        print(end='.', flush=True)

    def maxi(self, population_size=128, generations=5):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='i', fitness=creator.FitnessMax)

        # Attribute generator
        toolbox = base.Toolbox()

        toolbox.register("individual", tools.initIterate, creator.Individual, self.genIndividual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evalOneMax)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=13)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("progress", self.progress)

        hof = tools.HallOfFame(1)
        with closing(Pool()) as P:
            toolbox.register("map", P.map)
            pop_list, log_list = eaSimple(population_size, toolbox, cxpb=0.5, mutpb=0.2,
                                          ngen=generations, stats=stats, halloffame=hof, verbose=False)

        del creator.FitnessMax
        del creator.Individual

        self.pop = pd.DataFrame(pop_list, columns=inspect.getfullargspec(self.target).args)
        self.log = pd.DataFrame(log_list)

        return dict(zip(self.args, tuple(hof.items[0])))
