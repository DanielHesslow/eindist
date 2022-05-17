import torch.distributed as dist
import torch

from eindist.parsing import ParsedExpression
from einops import rearrange
import warnings

class EindistError(RuntimeError):
    """ Runtime error thrown by einops """
    pass

class Distributed:
    def __init__(self, process_group, dry_run=False, world_size=None, rank=None):
        self.dry_run = dry_run
        if not dry_run:
            self.pg = process_group
            if world_size is not None or rank is not None:
                warnings.warn("world size and rank should only be specified when using dry run. These values will be ignored.")
            self.world_size = dist.get_world_size(process_group) 
            self.rank      = dist.get_rank(process_group)
        else:
            if world_size == None or rank == None:
                raise Exception("world_size and rank must be specified when using dry_run")
            self.world_size = world_size
            self.rank = rank
            self.pg = None

def _gather(input_, pg):
    """Gather tensors and concatinate along the last dimension."""
    # Bypass the function if we are using only 1 GPU.
    if pg.world_size==1:
        return input_

    if pg.dry_run:
        tensor_list = [torch.zeros_like(input_) for _ in range(pg.world_size)]
        tensor_list[pg.rank] = input_
        return tensor_list

    tensor_list = [torch.empty_like(input_) for _ in range(pg.world_size)]
    tensor_list[pg.rank] = input_

    
    torch.distributed.all_gather(tensor_list, input_, group=pg.pg)
    return tensor_list

def all_to_all(output, input, group, async_op=False):
    # only nccl supports all_to_all for now.
    # and we can only do it if we have the same process group.
    
    world_size = dist.get_world_size(group)
    if dist.get_backend == "nccl" :
        dist.all_to_all(output, input, group = group, async_op=async_op)
    else:
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        for i in range(world_size):
            dist.scatter(output[i], input if i == rank else [], src = i, async_op=async_op)


def _reduce(input_, pg):
    """All-reduce the the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if pg.world_size == 1 or pg.dry_run:
        return input_

    # All-reduce.
    input_ = input_.clone() #for simplicity for now.
    torch.distributed.all_reduce(input_, group=pg.pg)
    return input_

def distribute_(pattern, tensor, dry = False, **kwargs):
    left_str, right_str = pattern.split("->")
    
    left = ParsedExpression(left_str)
    right = ParsedExpression(right_str)
    
    if ',' in pattern:
        raise EindistError("distribute should have one input and one output, not multiple.")
    
    diff = left.identifiers.symmetric_difference(right.identifiers)
    
    for k in diff:
        if not isinstance(kwargs[k], Distributed):
            raise EindistError("distribute will only allow modifications to indices that are distributed")
    
    # extract the relevant identifiers (where something happens)
    right_idents = right.identifiers.difference(left.identifiers)
    left_idents = left.identifiers.difference(right.identifiers)
    
    
    if len(right_idents) > 1:
        raise EindistError("distribute currently only works with distributions over single process groups")

    if len(left_idents) > 1:
        raise EindistError("distribute currently only works with distributions over single process groups")

    
    left_ident  = next(iter(left_idents))  if left_idents  else ""
    right_ident = next(iter(right_idents)) if right_idents else ""

    # This will trip up with lone distributed indices in superfluous parenthesis. @Fixme
    local_str  = left_str.replace(left_ident, "")
    local_str_r  = right_str.replace(right_ident, "")

    local_parsed = ParsedExpression(local_str)
    local_parsed_r = ParsedExpression(local_str_r)
    equal = local_parsed.composition == local_parsed_r.composition
    if not equal:
        raise EindistError(f"Right hand side and left hand side of expression must be equal apart from distributed indices. Got {local_str}, {local_str_r}")
    
    is_composite = dict() # only valid for new idents
    for composite_axis in right.composition + left.composition:
        for axis_name in composite_axis:
            is_composite[axis_name] = len(composite_axis)>1
    
    def handle_left(tensor):
        if is_composite[left_ident]:
            tensor_list = _gather(tensor, kwargs[left_ident])
            pattern = f"{left_ident} {local_str} -> {left_str}"
            return rearrange(tensor_list, pattern)
        else:
            return _reduce(tensor, kwargs[left_ident])

    def handle_right(tensor):
        if is_composite[right_ident]:
            world_size = kwargs[right_ident].world_size
            rank = kwargs[right_ident].rank
            pattern = f"{right_str} -> {right_ident} {local_str}"
            return rearrange(tensor, pattern, **{right_ident: world_size})[rank]
        else:
            return tensor

    if len(left_idents) == 1 and len(right_idents)==1: #fast-path for input + composite output + with all-to-all communication
        pg_out = kwargs[right_ident]
        pg_in = kwargs[left_ident]
            
        if is_composite[right_ident] and pg_out == pg_in and not pg_in.dry_run:
            pattern     = f"{right_str} -> {right_ident} {local_str}"

            world_size = kwargs[right_ident].world_size
            #.contiguous() is needed. not sure why tho?
            tensor = rearrange(tensor, pattern, **{right_ident: world_size}).contiguous()
            output = torch.empty_like(tensor)

            all_to_all(list(output), list(tensor), pg_in.pg)
            if is_composite[left_ident]:
                pattern = f"{left_ident} {local_str} -> {left_str}"
                return rearrange(output, pattern)
            else:
                return output.sum(0)

    if len(left_idents) == 1:
        tensor = handle_left(tensor)

    if len(right_idents)==1:
        tensor = handle_right(tensor)
    return tensor

class DistributeAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pattern, tensor, kwargs):
        ctx.pattern = pattern
        ctx.kwargs = kwargs
        return distribute_(pattern=pattern, tensor=tensor, **kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        # backward is just equal to reversing the expression, very nice :) 
        left, right = ctx.pattern.split("->")
        pattern = f"{right} -> {left}"
        return None, distribute_(pattern=pattern, tensor=grad_output, **ctx.kwargs), None

def distribute(pattern, tensor, **kwargs):
    # torch.autograd.Function doesn't like named arguments so we send them as a dict and unpack them in the forward.
    return DistributeAutogradFunction.apply(pattern, tensor, kwargs)
