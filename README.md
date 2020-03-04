# Tensorflow Foxhole Solver
A script that uses Tensorflow to solve the foxhole puzzle.

Details here: https://gurmeet.net/puzzles/fox-in-a-hole/

The main idea is that you can represent the fox's paths over time as a k-partite flow network.
The goal then becomes to find a vertex cut with 1 vertex from each independed vertex set.

By building a graph of connections between foxholes over time and the flow of "foxness" from beginning to end, we can represent the likelihood that you will check a foxhole by a given weight and optimize all weights by the amount of "foxness" that reaches the end of the time sequence.

