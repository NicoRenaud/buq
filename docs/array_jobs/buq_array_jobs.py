import numpy as np
import csv
from buq.kernels.rbf import RBFKernel
from buq.kernels.matern import MaternKernel
from buq.data.md import AlanineDipeptideMD
from buq.data.md import ChemicalReaction
from buq.buq import ClassicalBayesianQuadratureMD
import pickle


def get_config(file_name, id):
    # Read params.csv
    with open(file_name, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[id]

def get_target_function(name):
    if name == 'alanine':
        return AlanineDipeptideMD(mode='debug')
    elif name == 'reaction':
        return ChemicalReaction(mode='debug')
    else:
        raise ValueError(f"Unknown target function: {name}")


def get_kernel(name, lengthscale=0.75, noise=0.0):
    if name == 'rbf':
        return RBFKernel(lengthscale=lengthscale, noise=noise)
    elif name == 'matern':
        return MaternKernel(lengthscale=lengthscale, noise=noise)
    else:
        raise ValueError(f"Unknown kernel: {name}")


if __name__ == "__main__":

    import pickle
    from argparse import ArgumentParser


    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default='params.csv')
    parser.add_argument("--job_id", type=int, default=0)
    args = parser.parse_args()

    row = get_config(args.config, args.job_id)

    target = row["target"]
    lengthscale = (float(row["lengthscale"]))
    acq_function = row["acq_function"]
    kernel_type = row["kernel_type"]
    num_stats = int(row["num_stats"])
    num_init = int(row["num_init_samples"])

    results = []
    for _ in range(num_stats):

        target_function = get_target_function(target)
        rbf = get_kernel(kernel_type, lengthscale=lengthscale)

        x_data = target_function.generate_samples(num_init)
        y_data = target_function(x_data)

        bq = ClassicalBayesianQuadratureMD(rbf, target_function)
        res = bq.run(100, plot_fit=False)
        results.append(res)

    
    name = f"{target}_{kernel_type}_{lengthscale}_{acq_function}.pkl"
    with open(name,'wb') as f:
        pickle.dump(results, f)