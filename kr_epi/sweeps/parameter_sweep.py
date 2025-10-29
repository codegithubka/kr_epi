# kr_epi/sweeps/parameter_sweep.py
from __future__ import annotations
from typing import (
    Callable, Dict, Any, List, Optional, Tuple, Iterable, TYPE_CHECKING
)
import numpy as np
import pandas as pd
import itertools

if TYPE_CHECKING:
    from kr_epi.models.base import ODEBase
    from kr_epi.models.reactions import ReactionSystem
    from kr_epi.sweeps.runners import EnsembleResult

# Define a generic type for the result of a simulation
SimResult = Tuple[np.ndarray, np.ndarray] | EnsembleResult

def run_parameter_sweep(
    model_factory: Callable[..., ODEBase | ReactionSystem],
    sweep_parameters: Dict[str, Iterable],
    metric_function: Callable[[SimResult], float | Dict[str, float]],
    run_function: Callable[[ODEBase | ReactionSystem], SimResult],
    *,
    fixed_parameters: Optional[Dict[str, Any]] = None,
    n_runs: Optional[int] = None,
    seed_base: Optional[int] = None
) -> pd.DataFrame:
    """
    Runs simulations over a grid of parameters and collects metrics.

    This function handles the boilerplate of iterating through all
    parameter combinations, running simulations, and collecting results.

    Parameters
    ----------
    model_factory : Callable
        The constructor or factory function for the model
        (e.g., `SIRDemographyCounts` or `sir_demography_counts_reactions`).
    sweep_parameters : Dict[str, Iterable]
        A dictionary where keys are parameter names (e.g., 'beta') and
        values are iterables (e.g., np.linspace(0.1, 1.0, 10)).
    metric_function : Callable
        A function that takes the simulation result (e.g., (t, y) or
        EnsembleResult) and returns a scalar value or a dictionary
        of named metrics (e.g., lambda res: np.max(res[1][1, :])).
    run_function : Callable
        A function that takes the instantiated model/system and runs it.
        (e.g., lambda m: m.integrate(y0, t_span, t_eval=t))
        (e.g., lambda s: run_ensemble(Direct(s), n, x0, t_max, t_eval))
    fixed_parameters : Optional[Dict[str, Any]], optional
        A dictionary of parameters that are held constant for all runs.
    n_runs : Optional[int], optional
        (For stochastic sweeps) If passed, this is added to the parameter
        combinations for each run, typically used for seeding.
    seed_base : Optional[int], optional
        (For stochastic sweeps) Base seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row is a simulation run. Columns include
        all swept parameters and the metrics returned by metric_function.
    """
    if fixed_parameters is None:
        fixed_parameters = {}

    # Get the names of the parameters to sweep
    param_names = list(sweep_parameters.keys())
    # Get the iterables of values
    param_values = list(sweep_parameters.values())

    # Create all combinations of parameters
    param_combinations = list(itertools.product(*param_values))

    results_list = []

    # Check if this is a stochastic sweep with multiple runs per param set
    is_stochastic_sweep = n_runs is not None and n_runs > 0

    run_counter = 0 # For seeding

    for combo in param_combinations:
        # Create the parameter dictionary for this combination
        current_sweep_params = dict(zip(param_names, combo))
        
        # Merge fixed and swept parameters
        all_params = {**fixed_parameters, **current_sweep_params}
        
        if is_stochastic_sweep:
            # Run n_runs times for this single parameter combination
            for i in range(n_runs):
                run_params_dict = all_params.copy()
                run_params_dict['run_index'] = i
                
                # Create a unique seed for this specific run
                if seed_base is not None:
                    # This seed changes for *every* individual run
                    current_seed = seed_base + run_counter
                    run_params_dict['seed'] = current_seed
                    # We must pass the seed to the model factory if it accepts it
                    # This is tricky. Let's assume the run_function handles seeding.
                    # A better way: pass seed to run_function
                
                try:
                    # 1. Create model/system with current params
                    # We pass all params, factory will ignore ones it doesn't need
                    model_or_system = model_factory(**all_params)
                    
                    # 2. Run the simulation
                    # The run_function must be defined to handle the seed
                    # e.g., lambda s: Direct(s, seed=current_seed).run(...)
                    # This is too complex.
                    
                    # Simpler: run_ensemble handles the seeding.
                    # The user's run_function should be:
                    # lambda s: run_ensemble(Direct(s, seed_base), n_runs, ...)
                    # This means this loop shouldn't iterate n_runs.
                    
                    # --- Let's restart this logic ---
                    # The user decides whether to run 1 ODE or an ensemble
                    # by *what they pass* as run_function and metric_function.
                    
                    # 1. Create model/system
                    model_or_system = model_factory(**all_params)
                    
                    # 2. Run simulation (this might be a single ODE or a full ensemble)
                    sim_result = run_function(model_or_system)
                    
                    # 3. Calculate metrics (this might be np.max or extinction_prob)
                    metrics = metric_function(sim_result)
                    
                    # Store results
                    if isinstance(metrics, dict):
                        results_list.append({**current_sweep_params, **metrics})
                    else:
                        results_list.append({**current_sweep_params, "metric": metrics})
                        
                except Exception as e:
                    print(f"Error running combination {current_sweep_params}: {e}")
                    results_list.append({**current_sweep_params, "metric": np.nan})

        else: # This is a deterministic (or single stochastic) run
            try:
                # 1. Create model
                model_or_system = model_factory(**all_params)
                # 2. Run simulation
                sim_result = run_function(model_or_system)
                # 3. Calculate metrics
                metrics = metric_function(sim_result)
                
                # Store results
                if isinstance(metrics, dict):
                    results_list.append({**current_sweep_params, **metrics})
                else:
                    results_list.append({**current_sweep_params, "metric": metrics})
                    
            except Exception as e:
                print(f"Error running combination {current_sweep_params}: {e}")
                results_list.append({**current_sweep_params, "metric": np.nan})


    return pd.DataFrame(results_list)