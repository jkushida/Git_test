"""Differential evolution optimizer implementation.

This module provides a simple implementation of the differential evolution
algorithm for continuous optimisation problems.  The implementation follows the
classic DE/rand/1/bin strategy and is designed to be easy to read and reuse in
small experiments or teaching materials.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np


ObjectiveFunction = Callable[[np.ndarray], float]


@dataclass
class DEConfig:
    """Configuration parameters for the differential evolution algorithm."""

    bounds: Sequence[Tuple[float, float]]
    population_size: int = 20
    mutation_factor: float = 0.8
    crossover_probability: float = 0.7
    max_generations: int = 1000
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.mutation_factor:
            raise ValueError("mutation_factor must be positive")
        if not 0.0 <= self.crossover_probability <= 1.0:
            raise ValueError("crossover_probability must be in [0, 1]")
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if self.max_generations <= 0:
            raise ValueError("max_generations must be positive")
        if len(self.bounds) == 0:
            raise ValueError("bounds must contain at least one dimension")


@dataclass
class DEResult:
    """The result of a differential evolution optimisation run."""

    best_vector: np.ndarray
    best_score: float
    history: List[float]


def _initial_population(config: DEConfig, rng: np.random.Generator) -> np.ndarray:
    """Draw the initial population uniformly from the provided bounds."""

    lower_bounds = np.array([lo for lo, _ in config.bounds])
    upper_bounds = np.array([hi for _, hi in config.bounds])
    span = upper_bounds - lower_bounds
    population = rng.random((config.population_size, len(config.bounds)))
    population = lower_bounds + population * span
    return population


def differential_evolution(func: ObjectiveFunction, config: DEConfig) -> DEResult:
    """Perform differential evolution optimisation.

    Parameters
    ----------
    func:
        Callable that accepts a one-dimensional :class:`numpy.ndarray` and
        returns a scalar fitness value (lower is better).
    config:
        :class:`DEConfig` instance controlling the optimisation run.

    Returns
    -------
    :class:`DEResult`
        Contains the best solution vector, its score, and the best score after
        each generation.
    """

    rng = np.random.default_rng(config.seed)
    population = _initial_population(config, rng)
    scores = np.apply_along_axis(func, 1, population)

    best_idx = int(np.argmin(scores))
    best = population[best_idx].copy()
    best_score = float(scores[best_idx])
    history: List[float] = [best_score]

    lower_bounds = np.array([lo for lo, _ in config.bounds])
    upper_bounds = np.array([hi for _, hi in config.bounds])

    for _ in range(config.max_generations):
        for idx in range(config.population_size):
            # Mutation: select three distinct individuals different from idx.
            indices = list(range(config.population_size))
            indices.remove(idx)
            a_idx, b_idx, c_idx = rng.choice(indices, size=3, replace=False)
            a, b, c = population[a_idx], population[b_idx], population[c_idx]

            mutant = a + config.mutation_factor * (b - c)
            mutant = np.clip(mutant, lower_bounds, upper_bounds)

            # Binomial crossover.
            crossover = rng.random(len(config.bounds)) < config.crossover_probability
            if not np.any(crossover):
                crossover[rng.integers(len(config.bounds))] = True
            trial = np.where(crossover, mutant, population[idx])

            trial_score = float(func(trial))
            if trial_score <= scores[idx]:
                population[idx] = trial
                scores[idx] = trial_score

                if trial_score < best_score:
                    best = trial.copy()
                    best_score = trial_score
        history.append(best_score)

    return DEResult(best_vector=best, best_score=best_score, history=history)


def _rastrigin(x: np.ndarray) -> float:
    """Rastrigin test function (global minimum at the origin)."""

    A = 10.0
    return A * len(x) + np.sum(x * x - A * np.cos(2 * np.pi * x))


def _main() -> None:
    bounds = [(-5.12, 5.12)] * 2
    config = DEConfig(bounds=bounds, population_size=40, max_generations=200, seed=42)
    result = differential_evolution(_rastrigin, config)
    print("Best solution:", result.best_vector)
    print("Best score:", result.best_score)


if __name__ == "__main__":
    _main()
