# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from operator import attrgetter
from typing import List, Union

import torch
import torch.nn as nn


def efficient_conv_bn_eval_forward(bn: nn.modules.batchnorm._BatchNorm,
                                   conv: nn.modules.conv._ConvNd,
                                   x: torch.Tensor):
    """Code borrowed from mmcv 2.0.1, so that this feature can be used for old
    mmcv versions.

    Implementation based on https://arxiv.org/abs/2305.11624
    "Tune-Mode ConvBN Blocks For Efficient Transfer Learning"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for training as well. It reduces memory and computation cost.
    Args:
        bn (_BatchNorm): a BatchNorm module.
        conv (nn._ConvNd): a conv module
        x (torch.Tensor): Input feature map.
    """
    # These lines of code are designed to deal with various cases
    # like bn without affine transform, and conv without bias
    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = torch.zeros_like(bn.running_var)

    if bn.weight is not None:
        bn_weight = bn.weight
    else:
        bn_weight = torch.ones_like(bn.running_var)

    if bn.bias is not None:
        bn_bias = bn.bias
    else:
        bn_bias = torch.zeros_like(bn.running_var)

    # shape of [C_out, 1, 1, 1] in Conv2d
    weight_coeff = torch.rsqrt(bn.running_var +
                               bn.eps).reshape([-1] + [1] *
                                               (len(conv.weight.shape) - 1))
    # shape of [C_out, 1, 1, 1] in Conv2d
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # shape of [C_out, C_in, k, k] in Conv2d
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    # shape of [C_out] in Conv2d
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() *\
        (bias_on_the_fly - bn.running_mean)

    return conv._conv_forward(x, weight_on_the_fly, bias_on_the_fly)


def efficient_conv_bn_eval_control(bn: nn.modules.batchnorm._BatchNorm,
                                   conv: nn.modules.conv._ConvNd,
                                   x: torch.Tensor):
    """This function controls whether to use `efficient_conv_bn_eval_forward`.

    If the following `bn` is in `eval` mode, then we turn on the special
    `efficient_conv_bn_eval_forward`.
    """
    if not bn.training:
        # bn in eval mode
        output = efficient_conv_bn_eval_forward(bn, conv, x)
        return output
    else:
        conv_out = conv._conv_forward(x, conv.weight, conv.bias)
        return bn(conv_out)


def efficient_conv_bn_eval_graph_transform(fx_model):
    """Find consecutive conv+bn calls in the graph, inplace modify the graph
    with the fused operation."""
    modules = dict(fx_model.named_modules())

    patterns = [(torch.nn.modules.conv._ConvNd,
                 torch.nn.modules.batchnorm._BatchNorm)]

    pairs = []
    # Iterate through nodes in the graph to find ConvBN blocks
    for node in fx_model.graph.nodes:
        # If our current node isn't calling a Module then we can ignore it.
        if node.op != 'call_module':
            continue
        target_module = modules[node.target]
        found_pair = False
        for conv_class, bn_class in patterns:
            if isinstance(target_module, bn_class):
                source_module = modules[node.args[0].target]
                if isinstance(source_module, conv_class):
                    found_pair = True
        # Not a conv-BN pattern or output of conv is used by other nodes
        if not found_pair or len(node.args[0].users) > 1:
            continue

        # Find a pair of conv and bn computation nodes to optimize
        conv_node = node.args[0]
        bn_node = node
        pairs.append([conv_node, bn_node])

    for conv_node, bn_node in pairs:
        # set insertion point
        fx_model.graph.inserting_before(conv_node)
        # create `get_attr` node to access modules
        # note that we directly call `create_node` to fill the `name`
        # argument. `fx_model.graph.get_attr` and
        # `fx_model.graph.call_function` does not allow the `name` argument.
        conv_get_node = fx_model.graph.create_node(
            op='get_attr', target=conv_node.target, name='get_conv')
        bn_get_node = fx_model.graph.create_node(
            op='get_attr', target=bn_node.target, name='get_bn')
        # prepare args for the fused function
        args = (bn_get_node, conv_get_node, conv_node.args[0])
        # create a new node
        new_node = fx_model.graph.create_node(
            op='call_function',
            target=efficient_conv_bn_eval_control,
            args=args,
            name='efficient_conv_bn_eval')
        # this node replaces the original conv + bn, and therefore
        # should replace the uses of bn_node
        bn_node.replace_all_uses_with(new_node)
        # take care of the deletion order:
        # delete bn_node first, and then conv_node
        fx_model.graph.erase_node(bn_node)
        fx_model.graph.erase_node(conv_node)

    # regenerate the code
    fx_model.graph.lint()
    fx_model.recompile()


def turn_on_efficient_conv_bn_eval_for_single_model(model: torch.nn.Module):
    import torch.fx as fx

    # currently we use `fx.symbolic_trace` to trace models.
    # in the future, we might turn to pytorch 2.0 compile infrastructure to
    # get the `fx.GraphModule` IR. Nonetheless, the graph transform function
    # can remain unchanged. We just need to change the way
    # we get `fx.GraphModule`.
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    efficient_conv_bn_eval_graph_transform(fx_model)
    model.forward = fx_model.forward


def turn_on_efficient_conv_bn_eval(model: torch.nn.Module,
                                   modules: Union[List[str], str]):
    if isinstance(modules, str):
        modules = [modules]
    for module_name in modules:
        module = attrgetter(module_name)(model)
        turn_on_efficient_conv_bn_eval_for_single_model(module)
