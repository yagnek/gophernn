package main

import (
	"fmt"
	"math"

	"github.com/yagnek/gophernn/gnn"
)

func main() {
	activation := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

	cases := [][][]float64{
		{{0, 0}},
		{{1, 0}},
		{{0, 1}},
		{{1, 1}},
	}

	targets := [][][]float64{
		{{0}},
		{{1}},
		{{1}},
		{{0}},
	}

	nn, _ := gnn.NewNeuralNet([]int{2, 8, 4, 1}, 0.1, activation)

	fmt.Println("XOR NN!")
	fmt.Println("Before trainig:")
	fmt.Println("Resaults:")
	for i := range len(cases) {
		fmt.Printf("%.0f xor %.0f = %.0f\n", cases[i][0][0], cases[i][0][1], math.Round(nn.Query(cases[i])[0][0]))
	}

	epochs := 10000
	for e := 0; e < epochs; e++ {
		for i := 0; i < len(cases); i++ {
			nn.Train(cases[i], targets[i])
		}
	}
	fmt.Println("Resaults:")
	fmt.Println("After training:")
	for i := range len(cases) {
		fmt.Printf("%.0f xor %.0f = %.0f\n", cases[i][0][0], cases[i][0][1], math.Round(nn.Query(cases[i])[0][0]))
	}

}
