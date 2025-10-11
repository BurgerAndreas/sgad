

```bash
uv run scripts/train.py experiment=t1x_gad
```

- write a config file
- use the hip energy as in @examples/example_hip.py
- use T1x dataset as in @examples/example_t1x.py
- initialize buffer with transition states or reactant+product (equilibrium) geometries. optionally add noise to the geometries to encourage diversity.
- add automatic ckpt naming from ../hip/hip/logging_utils.py
- add frequency analysis to evaluation

- remove torsion from the code
- remove rdkit from the code

- before: sample = graph
- now: sample = list of atoms

Where do we need the graph?
use the graph creation function from the potential
- ajoint sampling computes graph when creating the dataloader, adjoint sampling focus on fixed bonds (conformers) -> graph does not change during noising or SDE integration
- graph only needs to be built in the outer loop when calling populate_buffer_with_samples_and_energy_gradients and integrate_sde (before we are calling the energy function)
- adjoint sampling is using a fully connected graph to make it easier to autodiff through the energy function, which we do not need
- change get_atomic_graph to use energy_model to create the graph
- Does EGNN controller take in bonds / graph edge_index?

- looking at Memo's examples, do we need to Eckart-project the Hessian before computing the eigenvector for GAD?


We will use a different energy function that does not use bonds. We will use a dataset without smiles, but a 3d pointcloud of atoms. We will do validation by frequency analysis.
We are not interested in conformers.

We will implement the energy function and frequency analysis later.

Do not modify EGNN unless necessary.

