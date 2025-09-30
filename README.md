# SNN text-to-text
This repo is a free time studying/exploring/playing with SNN's as a text generation from prompt.
Basic pipline is:
text -> discrete signal -> NN -> discrete signal -> text

## main.py
Input word outputs same word from trained word list
![alt text](https://github.com/Artur-Panasiuk/SNN_text_to_neurons_to_text/blob/main/learning_curve.png "learning curve")
![alt text](https://github.com/Artur-Panasiuk/SNN_text_to_neurons_to_text/blob/main/clusters.png "neuron clusters")
## main2.py
Input word outputs list of words from trained associations
=== Retrieval results ===
- cat    -> [('cat', 30), ('meow', 3), ('dog', 0), ('talk', 0)]
- dog    -> [('dog', 40), ('bark', 10), ('walk', 6), ('cat', 0)]
- tree   -> [('tree', 40), ('leaf', 9), ('talk', 3), ('walk', 2)]
- sky    -> [('sky', 10), ('blue', 5), ('clouds', 5), ('talk', 1)]
- talk   -> [('talk', 50), ('blue', 5), ('clouds', 5), ('walk', 4)]
