package main

import (
	"fmt"
	"math"

	"github.com/yagnek/gophernn/gnn"
)

func main() {
	activation := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

	cases := [][][]float64{}
	case1 := [][]float64{{0, 0}}
	case2 := [][]float64{{1, 0}}
	case3 := [][]float64{{0, 1}}
	case4 := [][]float64{{1, 1}}
	cases = append(cases, case1, case2, case3, case4)

	targets := [][][]float64{}
	target1 := [][]float64{{0}}
	target2 := [][]float64{{1}}
	target3 := [][]float64{{1}}
	target4 := [][]float64{{0}}
	targets = append(targets, target1, target2, target3, target4)

	nn, _ := gnn.NewNeuralNet([]int{2, 4, 1}, 0.1, activation)

	fmt.Println("XOR NN!")
	fmt.Println("Before trainig:")
	fmt.Println("Resaults:")
	for i := range len(cases) {
		fmt.Println(nn.Query(cases[i]))
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
		fmt.Println(nn.Query(cases[i]))
	}

}
