# Recurrent Networks

Why don’t we have modular networks?

In software there was a progression from monolith architectures (Google) to microservices (Uber/Netflix).

Can something similar occur in AI? A case study from the hippocampus:

Navigation in the hippocampus starts by identifying the location to providing a metric representation that enables continuous movement. 

A critical part of the system is the CA3 that has 3 competing tasks: retrieving previously known locations, identifying novel locations, and encoding the environment. This is handled by a recursion gradient in which upstream compuatation is defined by sparser activations, and downstream by dense recursion.

The output of sparse connectivity is an enegram that identifies a given known location. The output of recurrent connectivity (distal CA3) is an attractor that encodes the environment. Stability is achieved through competitive dynamics between plausible enegrams, informed by feedback cortical signals against the predictive representation in the attractor encoding.

The attractor network encodes, based on experience, “what is possible” within that environment. Is a multiverse. Which is propagated to the CA1 that selects the best choice based on memory, and past rewards. Such collapse of possibilities is again propagated to a metric space that encodes the best direction of movement.

## Why is that important for AI?  

Consider the task of navigation within a set of known (train) labyrinths. However, there is only partial observability. At every step, the network may gain additional information of the surroundings to help identify in which particular environment it is.

Now consider the training dynamics: reconstructing an environment, and navigating a labyrinth are both “easy” tasks. But training them together results in an exponential amount of data required. 

Far easier, is to feed the partial information to a first network, and use the output to navigate the labyrinth. One can even consider specialized “exploration” networks when the output from the first one is noisy. 

Is that relevant to agents? Absolutely. Consider the alternative between finetuning an LLM or training speacilized small models for substaks, and tool usage. Tomorrow I hope to publish the initial results!  

