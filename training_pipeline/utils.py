import azureml.core

from azureml.train.dnn import Gloo, Nccl, Mpi, ParameterServer
from azureml.train.hyperdrive import choice, randint, uniform, quniform, loguniform, qloguniform, normal, qnormal, lognormal, qlognormal
from azureml.train.hyperdrive import BanditPolicy, MedianStoppingPolicy, NoTerminationPolicy, TruncationSelectionPolicy
from azureml.train.hyperdrive import RandomParameterSampling, GridParameterSampling, BayesianParameterSampling
from azureml.exceptions import RunConfigurationException

def get_parameter_distribution(distribution, **kwargs):
    if "choice" in distribution.lower():
        parameter_distr = choice(
            kwargs.get("options", [])
        )
    elif "randint" in distribution.lower():
        parameter_distr = randint(
            upper=kwargs.get("upper", None)
        )
    elif "uniform" in distribution.lower():
        parameter_distr = uniform(
            min_value=kwargs.get("min_value", None),
            max_value=kwargs.get("max_value", None)
        )
    elif "quniform" in distribution.lower():
        parameter_distr = quniform(
            min_value=kwargs.get("min_value", None),
            max_value=kwargs.get("max_value", None),
            q=kwargs.get("q", None)
        )
    elif "loguniform" in distribution.lower():
        parameter_distr = loguniform(
            min_value=kwargs.get("min_value", None),
            max_value=kwargs.get("max_value", None),
        )
    elif "qloguniform" in distribution.lower():
        parameter_distr = qloguniform(
            min_value=kwargs.get("min_value", None),
            max_value=kwargs.get("max_value", None),
            q=kwargs.get("q", None)
        )
    elif "normal" in distribution.lower():
        parameter_distr = normal(
            mu=kwargs.get("mu", None),
            sigma=kwargs.get("sigma", None)
        )
    elif "qnormal" in distribution.lower():
        parameter_distr = qnormal(
            mu=kwargs.get("mu", None),
            sigma=kwargs.get("sigma", None),
            q=kwargs.get("q", None)
        )
    elif "lognormal" in distribution.lower():
        parameter_distr = lognormal(
            mu=kwargs.get("mu", None),
            sigma=kwargs.get("sigma", None)
        )
    elif "qlognormal" in distribution.lower():
        parameter_distr = qlognormal(
            mu=kwargs.get("mu", None),
            sigma=kwargs.get("sigma", None),
            q=kwargs.get("q", None)
        )
    else:
        parameter_distr = None
        raise RunConfigurationException(f"Parameter distribution for parameter not defined in settings. Please choose between \'choice\', \'randint\', \'uniform\', \'quniform\', \'loguniform\', \'qloguniform\', \'normal\', \'qnormal\', \'lognormal\' and \'qlognormal\'")
    return parameter_distr


def get_parameter_sampling(sampling_method, parameter_dict):
    if "random" in sampling_method.lower():
        ps = RandomParameterSampling(
            parameter_space=parameter_dict
        )
    elif "grid" in sampling_method.lower():
        ps = GridParameterSampling(
            parameter_space=parameter_dict
        )
    elif "bayesian" in sampling_method.lower():
        ps = BayesianParameterSampling(
            parameter_space=parameter_dict
        )
    else:
        ps = None
        raise RunConfigurationException("Parameter Sampling Method not defined in settings. Please choose between \'random\', \'grid\' and \'bayesian\'")
    return ps


def get_policy(policy_method, evaluation_interval, delay_evaluation, **kwargs):
    if "bandit" in policy_method.lower():
        policy = BanditPolicy(
            evaluation_interval=evaluation_interval,
            delay_evaluation=delay_evaluation,
            slack_factor=kwargs.get("slack_factor", None),
            slack_amount=kwargs.get("slack_amount", None)
        )
    elif "medianstopping" in policy_method.lower():
        policy = MedianStoppingPolicy(
            evaluation_interval=evaluation_interval,
            delay_evaluation=delay_evaluation
        )
    elif "notermination" in policy_method.lower():
        policy = NoTerminationPolicy()
    elif "truncationselection" in policy_method.lower():
        policy = TruncationSelectionPolicy(
            evaluation_interval=evaluation_interval,
            delay_evaluation=delay_evaluation,
            truncation_percentage=kwargs.get("truncation_percentage", None)
        )
    else:
        policy = NoTerminationPolicy()
    return policy


def get_distributed_backend(backend_name):
    if backend_name.lower() == "mpi":
        distrib_backend = Mpi()
    elif backend_name.lower() == "parameter_server":
        distrib_training_backend = ParameterServer()
    elif backend_name.lower() == "gloo":
        distrib_training_backend = Gloo()
    elif backend_name.lower() == "nccl":
        distrib_training_backend = Nccl()
    else:
        distrib_backend = None
    return distrib_backend
