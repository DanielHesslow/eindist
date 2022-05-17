# eindist
Differentiable einops-style wrapper over torch.distributed. 

## Install
    pip install git+https://github.com/DanielHesslow/eindist
## Example Usage

```python
    TP = Distributed(None) # None is default process group. Distribute across everyone

    x = torch.randn(4,4, requires_grad = True)

    y = distribute("x y -> x (TP y)", x,  TP = TP) # split x
    z = distribute("x (TP y) -> x y", y,  TP = TP) # gather

    # or..
    y = distribute("x y -> x (TP y)", x,  TP = TP) # split x
    z = distribute("TP x y -> x y",   y,  TP = TP) # reduce sum

    # or even..
    y = distribute("(SP x) y -> x (TP y)", x,  TP = TP, SP=TP) # all-to-all
    z = distribute("x (TP y) -> (SP x) y", x,  TP = TP, SP=TP) # all-to-all

```

## Why to use:

Simple, intuitive way to distribute workloads. Many different type of tensor parallelism is easy to implement.

And its all auto differentiable. 

## Why not to use 
Currently only single distributed index can be specified at each side of the expression. 

DistributedDataParallel is very good, when data parallel is enough, use it. 


