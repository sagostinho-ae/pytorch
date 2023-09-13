# Ensure torch.onnx.export respects nn.Module.forward API when using export_modules_as_functions

This PR addresses the problem described in #104880. In summary, when calling `torch.onnx.export` with `export_module_as_functions=True`,  the [CommonSubexpressionEliminator pass](https://github.com/pytorch/pytorch/blob/0433cb05961ce91474f09898220eb611570c107a/torch/onnx/utils.py#L609) (CSE) does not care from which scope a node belongs. This means that CSE is blind to the forward interface of the `nn.Module`s requested to be exported and thus runs ONNX function extraction takes place, certain functions will have more input nodes that compared to the corresponding `nn.Module`.

For a more in-depth explanation with images, I recommend reading https://github.com/pytorch/pytorch/issues/104880#issuecomment-1634483255.

How was the problem addressed in this PR. In two stages:

1. Identify node scopes that require special care based on the nn.Modules to export
2. Prevent CSE from replacing internally defined nodes in these special scopes, with nodes coming from external scopes.

I'll go into more detail into each of these steps below.

## Identifying Scopes of Interest

Imagine we have a graph composed of nodes with the following scopes

```
1. toplevel::/ScopeA
2. toplevel::/ScopeA
3. toplevel::/ScopeA::/ScopeAA
5. toplevel::/ScopeA::/ScopeAB
6. toplevel::/ScopeA::/ScopeAB
7. toplevel::/ScopeA::/ScopeAA::/ScopeAAA
8. toplevel::/ScopeA::/ScopeAB::/ScopeAAA
9. toplevel::/ScopeA::/ScopeB
10. toplevel::/ScopeA::/ScopeB
11. toplevel::/ScopeB
12. toplevel::/ScopeB
```

paired with the following `nn.Module`s `ScopeAB` and `ScopeB` that should be exported as ONNX functions.

For every node, CSE looks at the node type and variables and compares it against a set of nodes it has seen, to decide if it can replace them. In the example above, it can happen that node `1.` and node `9.` are an equivalent node, e.g., a `Constant` with value 0, and thus `9.` would be deleted in favor of using `1.`.

Let us now imagine node `10.` originally used `9.` as an input. After CSE, `10.` will start using `1.`.
Once we reach function extraction, because now `10.` makes reference to `1.`, a node in `toplevel::/ScopeA`, the  ONNX function representing `toplevel::/ScopeA::/ScopeB` will now have an additional input coming from outside of its scope, that is not present as a parameter in `ScopeB.forward()`.

In a nutshell, this is what this PR is fixing. *Preventing tensors that are not parameters is `Module.forward(self, param_a, param_b, ...)` (if `Module` is to be exported as an ONNX function), from creating additional inputs to the exported ONNX function.*

## Preventing CSE from replacing internally defined nodes in These Special Scopes

This PR proposes to treat the scopes of `nn.Module`s that need to be exported, differently. For the purpose of this explanation, let us call these scopes as "protected/special".

With this PR we **only** allow CSE to replace nodes in "protected" scopes with other nodes within the **same** scope. This is mostly targeting `Variable`s that are usually generated to populate the arguments of an expression/node. These often tend to be constants like the ones shown below:

```
%203 : int = prim::Constant[value=0](), scope: __main__.Network::/__main__.FixedShapeUnique::unique # <prefix>/pytorch/torch/functional.py:810:0

%204 : bool = prim::Constant[value=1](), scope: __main__.Network::/__main__.FixedShapeUnique::unique # <prefix>/pytorch/torch/functional.py:810:0

%205 : bool = prim::Constant[value=0](), scope: __main__.Network::/__main__.FixedShapeUnique::unique # <prefix>/pytorch/torch/functional.py:810:0
```


To achieve this, we need to:

1. Create independent [expression sets](https://github.com/pytorch/pytorch/blob/79c5e33349df10648ce586af118f09f2ccfd9c86/torch/csrc/jit/passes/common_subexpression_elimination.cpp#L26) for every scope that matches a module that needs to be exported. These sets coexist with the "global" scope node set (the only one that existed prior to this PR).
2. Assign each expression to a special scope, whenever their are enclosed in that scope, or to the global scope whenever
they are not enclosed by any special scope.
3. In order to ensure nodes are correctly assigned to each scope, deeper scopes need to be given priority during this assignment process.

How would the assignment look in the example above

```
1. toplevel::/ScopeA                        ->  global
2. toplevel::/ScopeA                        ->  global
3. toplevel::/ScopeA::/ScopeAA              ->  global
5. toplevel::/ScopeA::/ScopeAB              ->  toplevel::/ScopeA::/ScopeAB
6. toplevel::/ScopeA::/ScopeAB              ->  toplevel::/ScopeA::/ScopeAB
7. toplevel::/ScopeA::/ScopeAA::/ScopeAAA   ->  global
8. toplevel::/ScopeA::/ScopeAB::/ScopeAAA   ->  toplevel::/ScopeA::/ScopeAB
9. toplevel::/ScopeA::/ScopeB               ->  toplevel::/ScopeA::/ScopeB
10. toplevel::/ScopeA::/ScopeB              ->  toplevel::/ScopeA::/ScopeB
11. toplevel::/ScopeB                       ->  toplevel::/ScopeB
12. toplevel::/ScopeB                       ->  toplevel::/ScopeB
```

In order to ensure each node is assigned to the deepest scope possible, we need to produce a list of unique leaf scopes in descending order of depth, i.e.,

```
toplevel::/ScopeA::/ScopeAB
toplevel::/ScopeA::/ScopeB
toplevel::/ScopeB
```

If the node is not contained inside any of these scopes, it is assigned to the `global` scope.

# Need Some Help Bringing Things to the Finish Line

My current implementation achieves the desired outcome, but for the PR to be considered ready from my side, I would like to add tests to this implementation and this is where I could use some support.

