import pickle as pkl
import time
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from pathlib import Path
from qiskit import qpy
import dataclasses
from qiskit_aer import AerSimulator
import docplex.mp.model_reader
from qiskit_ibm_catalog import QiskitServerless

ROOT = Path(__file__).parent.parent
print(ROOT)

import sys
sys.path.append(str(ROOT))
from src.step_1 import get_cplex_sol, problem_mapping
from src.experiment import Experiment
from src.sbo.src.patterns.building_blocks.step_3 import HardwareExecutor

from doe import doe

def warm_start_nft(partial_file: str, exp_run: int, last_iter: int, instance: str, max_epoch: int | None = None):

    with open(partial_file, 'rb') as f:
        data = pkl.load(f)

    with open(Path(partial_file).with_name(f'{exp_run}_{last_iter}.pkl'), 'rb') as f:
        last_iter_info = pkl.load(f)
    
    lp_file = data['lp_file']
    experiment_id_with_exec = data['experiment_id']
    experiment_id = experiment_id_with_exec[:experiment_id_with_exec.index('/')]
    ansatz = data['ansatz']
    ansatz_params = data['ansatz_params']
    theta_initial = data['theta_initial']
    device = data['device']
    optimizer = data['optimizer']
    if max_epoch is None:
        max_epoch = data['max_epoch']
    alpha = data['alpha']
    shots = data['shots']
    theta_threshold = data['theta_threshold']
    refx = data['refx']
    refval = data['refvalue']

    # step 1 PROBLEM MAPPING
    obj_fn, ansatz_, _, backend, initial_layout = problem_mapping(lp_file.replace('.lp','-nocplexvars.lp'),
                                                               ansatz, ansatz_params, theta_initial, device, instance)
    theta_initial_ = data['step3_monitor_iter_thetas'][-1]

    # step_2 CIRCUIT OPTIMIZATION
    isa_ansatz_file = Path(lp_file).parent / f'{experiment_id}/isa_ansatz.qpy'
    with open(isa_ansatz_file, 'rb') as f:
        isa_ansatz = qpy.load(f)
        isa_ansatz = isa_ansatz[0]

    # step_3 HW EXECUTION
    out_file = Path(str(partial_file).replace('.pkl', '.1.pkl'))
    
    t = time.time()
    he = HardwareExecutor(
        objective_fun=obj_fn,
        backend=backend,
        isa_ansatz=isa_ansatz,
        optimizer_theta0=theta_initial_,
        optimizer_method=optimizer,
        refvalue=refval,
        sampler_options={'default_shots':shots, 'dynamical_decoupling': {'enable': True}}, # TODO: expose the entire sampler_options?
        use_session=False,
        iter_file_path_prefix=str(Path(lp_file).parent / experiment_id_with_exec),
        store_all_x=True,
        solver_options={"max_epoch": max_epoch, "alpha": alpha, 'theta_threshold': theta_threshold,
                        "epoch_start": last_iter_info['optimizer_internal_status_dict']['epoch'],
                        "idx_set_start": last_iter_info['optimizer_internal_status_dict']['idx_set'],
                        "step_start": last_iter_info['optimizer_internal_status_dict']['iter_in_epoch']+1}
    )

    he.optimization_monitor.calls_count = data['step3_fx_evals']
    he.optimization_monitor.callback_count = last_iter
    he.optimization_monitor.objective_monitor.best_x = data['step3_result_best_x']
    he.optimization_monitor.objective_monitor.best_fx = data['step3_result_best_fx']
    he.optimization_monitor.list_callback_inp = data['step3_monitor_iter_thetas']
    he.optimization_monitor.list_callback_res = data['step3_monitor_iter_gtheta']
    he.optimization_monitor.list_callback_monitor_best = data['step3_monitor_iter_best_fx']
    he.optimization_monitor.list_calls_inp = data['step3_monitor_calls_thetas']
    he.optimization_monitor.list_calls_res = data['step3_monitor_calls_gtheta']
    he.optimization_monitor.iter_best_x = data['step3_iter_best_x']
    he.optimization_monitor.iter_best_fx = data['step3_iter_best_fx']
    he.optimization_monitor.iter_fx_evals = data['step3_iter_fx_evals']

    result = he.run()
    step3_time = time.time() - t

    out = Experiment.from_step3(
        experiment_id_with_exec,
        ansatz, ansatz_params, theta_initial, device, optimizer, alpha, theta_threshold, lp_file, shots, refx, refval,
        Experiment.get_current_classical_hw(), (data['step3_time'] + step3_time) if data['step3_time'] is not None else None, data['step3_job_ids'] + he.job_ids,
        result, he.optimization_monitor
    )
    with open(out_file, 'bw') as f:
        pkl.dump(dataclasses.asdict(out), f)


def execute_multiple_runs(lp_file: str, experiment_id: str, num_exec: int, ansatz: str, ansatz_params: dict, 
                          theta_initial: str, device: str, instance: str, optimizer: str, max_epoch: int, alpha: float, shots: int,
                          run_on_serverless: bool, theta_threshold: float):

    # step 1 PROBLEM MAPPING
    obj_fn, ansatz_, theta_initial_, backend, initial_layout = problem_mapping(lp_file.replace('.lp','-nocplexvars.lp'),
                                                               ansatz, ansatz_params, theta_initial, device, instance)

    # create refval and check consistency of the obj fn
    refx, refval = get_cplex_sol(lp_file, obj_fn)

    (Path(lp_file).parent / experiment_id).mkdir(exist_ok=True)

    # step_2 CIRCUIT OPTIMIZATION
    isa_ansatz_file = Path(lp_file).parent / f'{experiment_id}/isa_ansatz.qpy'
    if isa_ansatz_file.is_file():
        with open(isa_ansatz_file, 'rb') as f:
            isa_ansatz = qpy.load(f)
            isa_ansatz = isa_ansatz[0]
    else:
        isa_ansatz = generate_preset_pass_manager(target=backend.target, optimization_level=3, initial_layout=initial_layout).run(ansatz_)
        with open(isa_ansatz_file, 'wb') as f:
            qpy.dump(isa_ansatz, f)

    # step_3 HW EXECUTION
    for exec in range(num_exec):
        experiment_id_with_exec = f'{experiment_id}/{exec}'
        if not run_on_serverless:
            out_file = Path(lp_file).parent / f'{experiment_id}/exp{exec}.pkl'
            if out_file.is_file():
                print(f'File {out_file} exists. Skipped.')
                continue
            
            t = time.time()
            he = HardwareExecutor(
                objective_fun=obj_fn,
                backend=backend,
                isa_ansatz=isa_ansatz,
                optimizer_theta0=theta_initial_,
                optimizer_method=optimizer,
                refvalue=refval,
                sampler_options={'default_shots':shots, 'dynamical_decoupling':{'enable': True}},
                use_session=False,
                iter_file_path_prefix=str(Path(lp_file).parent / experiment_id_with_exec),
                # verbose="iteration_all",
                store_all_x=True,
                solver_options={"max_epoch": max_epoch, "alpha": alpha, 'theta_threshold': theta_threshold},
            )
            result = he.run()
            step3_time = time.time() - t

            out = Experiment.from_step3(
                experiment_id_with_exec,
                ansatz, ansatz_params, theta_initial, device, optimizer, alpha, theta_threshold, lp_file, shots, refx, refval,
                Experiment.get_current_classical_hw(), step3_time, he.job_ids,
                result, he.optimization_monitor
            )
            with open(out_file, 'bw') as f:
                pkl.dump(dataclasses.asdict(out), f)

        else:
            serverless = QiskitServerless()
            remote_step3 = next(program for program in serverless.list() if program.title == "execute_on_hw")

            with open(lp_file.replace('.lp','-nocplexvars.lp'), "r", encoding="utf-8") as file:
                lp_content = file.read()

            job = remote_step3.run(
                experiment_id_with_exec = experiment_id_with_exec,
                lp_content = lp_content,
                isa_ansatz = isa_ansatz,
                theta_initial_ = theta_initial_,
                optimizer = optimizer,
                refval = refval,
                shots = shots,
                max_epoch = max_epoch,
                alpha = alpha,
                device = device,
                instance = instance,
            )
            print('Job ', job.job_id)


        # step 4 POSTPROCESSING
        # step 4 is deferred to another script


if __name__ == '__main__':
    # for _, exp in doe.items():
    #     if exp['device'] != 'AerSimulator':
    #         continue
    #     execute_multiple_runs(**exp, instance='', run_on_serverless=False)
    execute_multiple_runs(**doe['1/31bonds/bfcdR3rep_piby3_AerSimulator_0.2'], instance='', run_on_serverless=False)

