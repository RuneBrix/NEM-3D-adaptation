import os, time
import numpy as np
from PIL import Image
from pathlib import Path
import timm

# from quantus.functions.perturb_func import batch_uniform_noise
from quantus.functions.similarity_func import difference
from quantus.helpers.perturbation_utils import (
    make_changed_prediction_indices_func,
    make_perturb_func,
)
from .metric_util import gkern, CausalMetric, auc
from torch import nn
import torch
import copy
from quantus.metrics.randomisation.mprt import MPRT
import quantus


def gen_attr(method, model, train_data, test_data, exp_store, check_exist=False):
    save_path = Path(exp_store) / "attr"
    if check_exist & os.path.exists(save_path):
        return save_path
    print("Generating attributions")

    os.makedirs(save_path, exist_ok=True)
    start_time = time.time()
    attrs = method.gen_attrs(test_data)
    average_time = (time.time() - start_time) / len(test_data)
    for attr, _, name in attrs:
        attr = (((attr - attr.min()) / (attr.max() - attr.min())) * 255).astype(
            np.uint8
        )
        Image.fromarray(attr).save(save_path / f"{name}.png")
    # store metadata
    np.save(Path(exp_store) / "time.npy", np.array([average_time]))
    return save_path


def exp_faithfullness(
    method, model, train_data, test_data, exp_store, check_exist=True
):
    attrs_path = gen_attr(
        method, model, train_data, test_data, exp_store, check_exist=True
    )
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen // 2)
    prob_model = nn.Sequential(model, nn.Softmax(dim=1)).cuda().eval()
    res = []
    xs = []
    names = []
    attrs = []
    iter_data = iter(test_data)

    for attr_path in sorted(attrs_path.iterdir(), key=lambda f: int(f.stem)):
        name = attr_path.stem
        attr = np.array(Image.open(attr_path))
        X, y = next(iter_data)
        attrs.append(attr)
        xs.append(X)
        names.append(name)
    if (
        len(xs) == 0
    ):  # <-- if no attributions are generated (Honestly, this just tackled IBA for VITs...)
        return
    xs = torch.stack(xs).squeeze()
    attrs = np.array(attrs)

    with torch.no_grad():
        metric = CausalMetric(
            model=prob_model, mode="del", step=224, substrate_fn=torch.zeros_like
        )
        del_scores = metric.evaluate(img_batch=xs, exp_batch=attrs, batch_size=25)
        del_scores = [auc(del_scores[:, i]) for i in range(len(del_scores[0]))]

        metric = CausalMetric(model=prob_model, mode="ins", step=224, substrate_fn=blur)
        ins_scores = metric.evaluate(img_batch=xs, exp_batch=attrs, batch_size=25)
        ins_scores = [auc(ins_scores[:, i]) for i in range(len(ins_scores[0]))]

        values = [
            in_score - d_score for in_score, d_score in zip(ins_scores, del_scores)
        ]
    for name, value in zip(names, values):
        res.append((name, value))
    mean_res = np.mean([r[1] for r in res])
    print(f"faithfullness: {mean_res}")
    np.save(Path(exp_store) / "faithfullness.npy", np.array(res))
    return


###### BELOW CHANGES WERE NEEDED FOR IBA... PRETTY ANNOYING THAT DEEPCOPY IS NOT SAFE ########
def copy_timm_model(model):
    model_name = model.default_cfg.get("architecture", None)
    model_copy = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=model.num_classes,
        in_chans=model.default_cfg.get("in_chans", 3),
    )
    model_copy.load_state_dict(model.state_dict())
    device = next(model.parameters()).device
    model_copy.to(device)
    model_copy.eval()
    return model_copy


def get_random_layer_generator_override(self, order: str = "top_down", seed: int = 42):
    """
    In every iteration yields a copy of the model with one additional layer's parameters randomized.
    For cascading randomization, set order (str) to 'top_down'. For independent randomization,
    set it to 'independent'. For bottom-up order, set it to 'bottom_up'.

    Parameters
    ----------
    order: string
        The various ways that a model's weights of a layer can be randomised.
    seed: integer
        The seed of the random layer generator.

    Returns
    -------
    layer.name, random_layer_model: string, torch.nn
        The layer name and the model.
    """
    original_parameters = self.state_dict()
    random_layer_model = copy_timm_model(self.model)
    modules = [
        layer
        for layer in random_layer_model.named_modules()
        if (hasattr(layer[1], "reset_parameters"))
    ]

    if order == "top_down":
        modules = modules[::-1]

    for module in modules:
        if order == "independent":
            random_layer_model.load_state_dict(original_parameters)
        torch.manual_seed(seed=seed + 1)
        module[1].reset_parameters()
        yield module[0], random_layer_model


quantus.helpers.model.pytorch_model.PyTorchModel.get_random_layer_generator = (
    get_random_layer_generator_override
)


###### END OF ANNOYING CHANGES ########
def exp_randomization(
    method, model, train_data, test_data, exp_store, check_exist=True
):
    attrs_path = gen_attr(
        method, model, train_data, test_data, exp_store, check_exist=True
    )
    xs = []
    y_s = []
    names = []
    attrs = []
    iter_data = iter(test_data)

    for attr_path in sorted(attrs_path.iterdir(), key=lambda f: int(f.stem)):
        name = attr_path.stem
        attr = np.array(Image.open(attr_path))
        try:
            X, y = next(iter_data)
        except:
            break
        if len(X.unique()) == 1:
            continue
        attrs.append(attr)
        xs.append(X)
        y_s.append(y)
        names.append(name)
    if (
        len(xs) == 0
    ):  # <-- if no attributions are generated (Honestly, this just tackled IBA for VITs...)
        return
    x_batch = torch.stack(xs).squeeze().numpy()
    y_batch = torch.tensor(y_s).numpy()

    perturbed_model = copy_timm_model(model)

    attr_func = quantus_wrapper(train_data=train_data, method=method)
    try:
        q_metric = MPRT(
            display_progressbar=True,
            return_last_correlation=True,
            return_aggregate=True,
            normalise=True,
        )
        res = q_metric(
            model=perturbed_model,
            x_batch=x_batch,
            y_batch=y_batch,
            device="cuda",
            explain_func=attr_func,
        )
        print(res)
        mean_res = np.nanmean(res)
    except Exception as e:
        print(f"Randomization experiment failed with error: {e}")
        mean_res = 0
        res = [np.nan] * len(x_batch)
    print(f"Randomization: {mean_res}")
    np.save(Path(exp_store) / "randomization.npy", np.array(res))


class quantus_wrapper:
    def __init__(self, train_data, method):
        self.train_data = train_data
        self.method = method

    def __call__(self, model, inputs, targets, device="gpu"):
        self.method.reinit(model=model, train_data=self.train_data)
        return np.array(
            [
                r[0]
                for r in self.method.gen_attrs(
                    zip(torch.tensor(inputs).unsqueeze(1), torch.tensor(targets))
                )
            ]
        )


def exp_robustness(method, model, train_data, test_data, exp_store, check_exist=True):
    attrs_path = gen_attr(
        method, model, train_data, test_data, exp_store, check_exist=True
    )
    xs = []
    y_s = []
    names = []
    attrs = []
    samples = 30
    iter_data = iter(test_data)

    norm_numerator = fro_norm
    norm_denominator = fro_norm
    similarity_func = difference
    changed_prediction_indices = make_changed_prediction_indices_func(False)
    perturb_func = make_perturb_func(
        batch_uniform_noise,
        None,
        lower_bound=0.2,
        upper_bound=None,
    )

    for attr_path in sorted(attrs_path.iterdir(), key=lambda f: int(f.stem)):
        name = attr_path.stem
        attr = np.array(Image.open(attr_path))
        try:
            X, y = next(iter_data)
        except:
            break
        attrs.append(attr)
        xs.append(X)
        y_s.append(y)
        names.append(name)
    if (
        len(xs) == 0
    ):  # <-- if no attributions are generated (Honestly, this just tackled IBA for VITs...)
        return
    x_batch = torch.stack(xs).squeeze().numpy()
    y_batch = torch.tensor(y_s).numpy()
    a_batch = torch.tensor(np.array(attrs)).numpy()

    batch_size = x_batch.shape[0]

    a_batch = method.gen_attrs(
        zip(torch.tensor(x_batch).unsqueeze(1), torch.tensor(y_batch))
    )
    a_batch = [a[0] for a in a_batch]
    a_batch = np.array(a_batch)
    a_batch = a_batch.reshape(batch_size, -1)
    a_batch = (a_batch - a_batch.min(axis=-1, keepdims=True)) / (
        a_batch.max(axis=-1, keepdims=True) - a_batch.min(axis=-1, keepdims=True)
    )
    similarities = np.zeros((batch_size, samples)) * np.nan

    for step_id in range(samples):
        # Perturb input.
        x_perturbed = perturb_func(
            arr=x_batch.reshape(batch_size, -1),
            indices=np.tile(np.arange(0, x_batch[0].size), (batch_size, 1)),
        )
        x_perturbed = x_perturbed.reshape(*x_batch.shape)

        changed_prediction = changed_prediction_indices(model, x_batch, x_perturbed)
        # Generate explanation based on perturbed input x.
        a_perturbed = method.gen_attrs(
            zip(torch.tensor(x_perturbed).unsqueeze(1), torch.tensor(y_batch))
        )
        a_perturbed = [a[0] for a in a_perturbed]
        a_perturbed = np.array(a_perturbed)
        a_perturbed = a_perturbed.reshape(batch_size, -1)
        a_perturbed = (a_perturbed - a_perturbed.min(axis=-1, keepdims=True)) / (
            a_perturbed.max(axis=-1, keepdims=True)
            - a_perturbed.min(axis=-1, keepdims=True)
        )

        # Measure similarity.

        sensitivities = similarity_func(a=a_batch, b=a_perturbed)
        numerator = norm_numerator(a=sensitivities)
        denominator = norm_denominator(a=a_batch)
        similarities[:, step_id] = numerator / denominator
        similarities[changed_prediction, step_id] = np.nan

    mean_res = np.nanmean(similarities)
    print(f"Robustness: {mean_res}")
    np.save(Path(exp_store) / "robustness.npy", np.array(similarities))


def batch_uniform_noise(
    arr: np.array,
    indices: np.array,
    lower_bound: float = 0.02,
    upper_bound=None,
    **kwargs,
) -> np.array:
    """
    Add noise to the input at indices as sampled uniformly random from [-lower_bound, lower_bound].
    if upper_bound is None, and [lower_bound, upper_bound] otherwise.

    Parameters
    ----------
    arr: np.ndarray
         Array to be perturbed.
    indices: int, sequence, tuple
        Array-like, with a subset shape of arr.
    lower_bound: float
            The lower bound for uniform sampling.
    upper_bound: float, optional
            The upper bound for uniform sampling.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    arr_perturbed: np.ndarray
         The array which some of its indices have been perturbed.
    """

    # Assert dimensions
    assert len(arr.shape) == 2, (
        "The array must be 2-dimensional, first dimension corresponding to the batch size, and the second to the features"
    )
    assert len(indices.shape) == 2, (
        "The indices array must be 2-dimensional, first dimension corresponding to the batch size, and the second to the indices to perturb"
    )

    batch_size = arr.shape[0]
    arr_perturbed = copy.copy(arr)

    # Sample the noise.
    if upper_bound is None:
        noise = np.random.uniform(low=-lower_bound, high=lower_bound, size=arr.shape)
    else:
        assert upper_bound > lower_bound, (
            "Parameter 'upper_bound' needs to be larger than 'lower_bound', "
            "but {} <= {}".format(upper_bound, lower_bound)
        )
        noise = np.random.uniform(low=lower_bound, high=upper_bound, size=arr.shape)

    # Perturb the array.
    arr_perturbed[np.arange(batch_size)[:, None], indices] = (arr_perturbed + noise)[
        np.arange(batch_size)[:, None], indices
    ]

    return arr_perturbed


def fro_norm(a: np.array) -> float:
    """
    Calculate Frobenius norm for an array.

    Parameters
    ----------
    a: np.ndarray
         The array to calculate the Frobenius on. If 2D, the array is assumed to be batched.

    Returns
    -------
    float
        The norm.
    """
    assert a.ndim == 1 or a.ndim == 2, (
        "Check that 'fro_norm' receives a 1D or 2D array."
    )
    return np.linalg.norm(a, axis=-1)
