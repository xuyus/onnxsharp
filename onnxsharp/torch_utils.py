# Copyright 2024 XUYUS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import os
import numpy
from onnx import TensorProto, numpy_helper

global run_steps

run_steps = 0


def dump_parameters_and_grads_before_step_start(module, dir_to_save, max_step_run=2):
    global run_steps
    # def _post_backward_module_hook(module, inputs):
    print(f"enter dump_parameters_and_grads_before_step_start at step {run_steps}")
    # module.register_forward_pre_hook(_post_backward_module_hook)
    # def _ort_pre_forward_module_hook(module, inputs):

    dir_for_param = os.path.join(dir_to_save, "params")
    os.makedirs(dir_for_param, exist_ok=True)
    torch.save(module, os.path.join(dir_for_param, f"{run_steps}"))

    dir_for_grad = os.path.join(dir_to_save, "grads")
    os.makedirs(dir_for_grad, exist_ok=True)
    grads = {}
    for name, param in module.named_parameters():
        grads[name] = param.grad
    torch.save(grads, os.path.join(dir_for_grad, f"{run_steps}"))

    run_steps += 1
    if run_steps >= max_step_run:
        raise RuntimeError("Stop by intention to save the model.")


def compare_parameters_and_grads(a_dir_to_load, b_dir_to_load, step):
    # def _post_backward_module_hook(module, inputs):
    print(f"enter compare_parameters_and_grads at step {step}")
    # module.register_forward_pre_hook(_post_backward_module_hook)
    # def _ort_pre_forward_module_hook(module, inputs):

    a_params = torch.load(os.path.join(a_dir_to_load, "params", f"{step}"))
    b_params = torch.load(os.path.join(b_dir_to_load, "params", f"{step}"))

    b_param_map = {}
    for name, param in b_params.named_parameters():
        b_param_map[name] = param

    for name, param in a_params.named_parameters():
        non_close_count = torch.sum(
            torch.logical_not(
                torch.isclose(
                    param, b_param_map[name], rtol=1e-05, atol=1e-08, equal_nan=True
                )
            )
        )
        if non_close_count > 0:
            print(
                f"param [{name}, {param.shape}] has {non_close_count} non-close elements. e.g. {param.view(-1)[0:20]} vs {b_param_map[name].view(-1)[0:20]}"
            )
            # print(torch.nonzero(torch.logical_not(torch.isclose(param, b_param_map[name], rtol=1e-05, atol=1e-08, equal_nan=True))))

    a_grads = torch.load(os.path.join(a_dir_to_load, "grads", f"{step}"))
    b_grads = torch.load(os.path.join(b_dir_to_load, "grads", f"{step}"))

    for name, grad in a_grads.items():
        if grad is None and b_grads[name] is None:
            continue
        if grad is None or b_grads[name] is None:
            print(
                f"grad [{name}, {grad.shape}] has non-close elements. One is None, the other is not."
            )
            continue

        non_close_count = torch.sum(
            torch.logical_not(
                torch.isclose(
                    grad, b_grads[name], rtol=1e-05, atol=1e-08, equal_nan=True
                )
            )
        )
        if non_close_count > 0:
            print(
                f"grad [{name}, {grad.shape}] has {non_close_count} non-close elements. e.g. {grad.view(-1)[0:20]} vs {b_grads[name].view(-1)[0:20]}"
            )


from .npy_utils import npy_summurize_array


class DataObserver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, name, input):
        val = None
        if input is None or not isinstance(input, torch.Tensor):
            val = input
        else:
            val = input.detach().cpu().numpy()
        npy_summurize_array(val, name + "_forward")
        ctx.name = name
        return input.detach() if input is not None else None

    @staticmethod
    def backward(ctx, grad_output):
        val = None
        if grad_output is None or not isinstance(grad_output, torch.Tensor):
            val = grad_output
        else:
            val = grad_output.detach().cpu().numpy()

        npy_summurize_array(val, ctx.name + "_backward")
        return None, grad_output.detach() if grad_output is not None else None
