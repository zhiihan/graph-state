# ClusterSim

A cluster state simulator for measurement-based quantum computation, in browser.

> Based on the paper 'Fast simulation of stabilizer circuits using a graph state representation' by Simon Anders and Hans J. Briegel ([here](https://arxiv.org/abs/quant-ph/0504117v2))

## Installation

To install with pip, use:

`pip install -r requirements.txt`

To install with conda, make an empty conda environment, and then run:

`conda install --file requirements.txt`

After installation, run 

`python grid3dfigure.py`

### Graph State

* This implementation is based around graph states.
These were introduced in the paper about entanglement purification ([here](https://arxiv.org/abs/quant-ph/0512218)) to study the entanglement properties of certain multi-qubit systems
* Takes their name from graphs in maths
* Each qubit corresponds to a vertex of the graph, and each edge indicates which qubits have interacted.
* There is a bijection between stabilizer states (the states that can appear in a stabilizer circuit) and graph states. That is, every graph state has a corresponding stabilizer state, and every stabilizer state has a corresponding graph state.
* This can be shown as: Any stabilizer state can be transformed to a graph state by applying a tensor product of local Clifford operations. These are known as *vertex operators* (VOPs). See [this paper](https://arxiv.org/abs/quant-ph/0111080) and [this paper](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.69.022316).
* The standard approach is to store a tableau of stabillzer operators (an $n \times n$ matrix of Pauli operators).
* The improved algorithm needs only the graph state and the list of VOPs, and requires space $\mathcal{O}(n \log n)$.
* To then change the state, measurement is studied in [this paper](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.69.062311), and gate application in [the paper mentioned above](https://arxiv.org/pdf/quant-ph/0504117v2.pdf).
