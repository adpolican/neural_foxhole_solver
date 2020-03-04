# neural_foxhole_solver
A script that uses Tensorflow to solve the foxhole puzzle.
Details here: https://gurmeet.net/puzzles/fox-in-a-hole/

The main idea is that you can represent the fox's paths over time as a k-partite flow network.
The goal then becomes to find a vertex cut with 1 vertex from each independed vertex set.

