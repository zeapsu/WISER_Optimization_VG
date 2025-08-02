from pathlib import Path

ROOT = Path(__file__).parent.parent
print(ROOT)

import sys
sys.path.append(str(ROOT))

import time
import pickle as pkl
import dataclasses
import numpy as np
import docplex.mp.model
import docplex.mp.model_reader

from src.sbo.src.optimizer.local_search import repeated_local_search_general
from src.sbo.src.optimizer.optimization_monitor import uncompress_x, compress_x
from src.experiment import Experiment
from src.step_1 import problem_mapping, model_to_obj
from doe_localsearch import doe_localsearch

def postprocess_iter(iter_file, doe_ls, num_qubits, obj_fn, refval, outfile_tag):
    out_file = str(iter_file).replace('.pkl', f'_LS{outfile_tag}.pkl')

    if Path(out_file).exists():
        with open(out_file, 'rb') as f:
            data = pkl.load(f)
        
        idx_best_per_job = [np.argmin(a) for a in data['fx']]
        x_best_per_job = [data['x'][job_idx][idx] for job_idx, idx in enumerate(idx_best_per_job)]
        fx_best_per_job = [data['fx'][job_idx][idx] for job_idx, idx in enumerate(idx_best_per_job)]
        feval_iter = data['feval_iter']
        list_processed_x_iter = data['x']
        list_processed_fx_iter = data['fx']
        elapsed = data['time']
        return x_best_per_job, fx_best_per_job, feval_iter, list_processed_x_iter, list_processed_fx_iter, elapsed

    with open(iter_file, 'rb') as f:
        data = pkl.load(f)
    
    list_x = data['list_x']
    list_fx = data['list_fx']
    try:
        if len(np.array(list_x).flatten()) == 0:
            return [], [], [], [], [], 0
    except ValueError:
        pass

    if not isinstance(list_x[0], list): # backward compatibility
        list_x = [list_x]
    if not isinstance(list_fx[0], list): # backward compatibility
        list_fx = [list_fx]
    
    x_best_per_job = []
    fx_best_per_job = []
    feval_iter = []
    list_processed_x_iter = []
    list_processed_fx_iter = []

    t = time.time()
    for l_x, l_fx in zip(list_x, list_fx):
        # uncompress
        l_x = [uncompress_x(x, num_qubits) for x in l_x]
        
        x_best_job, fx_best_job = None, None
        feval_job = 0
        list_processed_x_job = []
        list_processed_fx_job = []
        for x, val in zip(l_x, l_fx):
            if x_best_job is None:
                x_best_job, fx_best_job = x, val
            x_best, val_best, num_epochs, num_fevals, vals = repeated_local_search_general(
                x, val, obj_fn, doe_ls | {'refval': refval}
            )

            list_processed_x_job.append(compress_x(x_best))
            list_processed_fx_job.append(val_best)
            feval_job += sum(num_fevals)

            if val_best < fx_best_job:
                x_best_job, fx_best_job = x_best, val_best
        
        list_processed_x_iter.append(list_processed_x_job)
        list_processed_fx_iter.append(list_processed_fx_job)
        x_best_per_job.append(x_best_job)
        fx_best_per_job.append(fx_best_job)
        feval_iter.append(feval_job)

    elapsed = time.time()-t
    if outfile_tag is not None:
        with open(out_file, 'wb') as f:
            pkl.dump({'x': list_processed_x_iter, 'fx': list_processed_fx_iter, 'time': elapsed, 'feval_iter': feval_iter}, f)

    return x_best_per_job, fx_best_per_job, feval_iter, list_processed_x_iter, list_processed_fx_iter, elapsed


if __name__ == "__main__":
        
        doe = doe_localsearch['unconstrained']
        # experiments: list[Experiment] = Experiment.read_experiments('1/31bonds/TwoLocal2rep_piby3_kyiv_0.1/')
        experiments: list[Experiment] = Experiment.read_experiments('1/31bonds/bfcd3rep_piby3_AerSimulator_0.2/') # change to bfcd

        for xp in experiments:
            if xp.has_step4():
                continue

            lp_file = xp.lp_file.replace('/home/gabriele-agliardi/data/Client-Vanguard-Optimization', str(ROOT))
            out_file = Path(lp_file).parent / f'{xp.experiment_id.replace("/", "/exp")}_LS{doe["local_search_doe"]}.pkl'
            
            if out_file.exists():
                continue
            
            with open(out_file, 'bw') as f: # placeholder file
                pkl.dump({}, f)

            # step 1
            # obj_fn, ansatz_, theta_initial_, backend, initial_layout = problem_mapping(lp_file.replace('.lp','-nocplexvars.lp'), xp.ansatz, xp.ansatz_params, xp.theta_initial, xp.device, '')

            model: docplex.mp.model.Model = docplex.mp.model_reader.ModelReader.read(lp_file.replace('.lp','-nocplexvars.lp'))

            obj_fn = model_to_obj(model)
            num_vars = model.number_of_binary_variables


            if 'local_search_maxfevals_per_variable' in doe:
                doe['local_search_maxfevals'] = num_vars * doe.pop('local_search_maxfevals_per_variable')

            # step 2 (not needed)

            # step 3 (not needed)

            # step 4

            iter_best_fx = []
            iter_fx_evals = []
            result_best_x, result_best_val = None, None
            step4_time = 0

            # pathlist = Path(xp.lp_file).parent.glob(f'{xp.experiment_id}_*.pkl')
            for iter in range(len(xp.step3_monitor_iter_best_fx)):
                print(xp.experiment_id, ' - iter ', iter)

                x_best_per_job, fx_best_per_job, feval_iter, list_processed_x_iter, list_processed_fx_iter, proc_time =\
                    postprocess_iter(Path(lp_file).parent / f'{xp.experiment_id}_{iter}.pkl', doe, num_vars, obj_fn, xp.refvalue, doe["local_search_doe"])
                
                if len(fx_best_per_job) == 0:
                    iter_best_fx.append(None)
                    iter_fx_evals.append(None)

                    step4_time += proc_time
                    continue

                best_iter = np.argmin(fx_best_per_job)
                if result_best_val is None or fx_best_per_job[best_iter] < result_best_val:
                    result_best_x, result_best_val = x_best_per_job[best_iter], fx_best_per_job[best_iter]

                iter_best_fx.append(fx_best_per_job)
                iter_fx_evals.append(feval_iter)

                step4_time += proc_time
            
            xp.local_search_doe = doe["local_search_doe"]
            xp.step4_time = step4_time
            xp.step4_num_epochs = None
            xp.step4_fx_evals = iter_fx_evals
            xp.step4_result_best_x = result_best_x
            xp.step4_result_best_fx = result_best_val
            xp.step4_iter_best_fx = iter_best_fx

            with open(out_file, 'bw') as f:
                pkl.dump(dataclasses.asdict(xp), f)

