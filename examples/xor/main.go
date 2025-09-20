package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/yagnek/gophernn/gnn"
)

type NeuralNet struct {
	Shape      []int
	LR         float64
	Activation func(x float64) float64
	Weights    []*gnn.Matrix
}

func InitRandom(m *gnn.Matrix) *gnn.Matrix {
	for i := range m.Rows {
		for j := range m.Cols {
			m.Data[i][j] = rand.Float64() - 0.5
		}
	}
	return m
}

func NewNeuralNet(shape []int, lr float64, activation func(x float64) float64) (nn *NeuralNet, err error) {
	weights := []*gnn.Matrix{}
	for i := range (len(shape)) - 1 {
		weights = append(weights, InitRandom(gnn.ZeroMatrix(shape[i+1], shape[i])))
	}
	return &NeuralNet{
		Shape:      shape,
		LR:         lr,
		Activation: activation,
		Weights:    weights,
	}, nil
}

func (nn *NeuralNet) Query(inputsList [][]float64) [][]float64 {
	inputs, _ := gnn.NewMatrix(inputsList)
	inputs = inputs.T()

	outputs := []*gnn.Matrix{}
	outputs = append(outputs, inputs)
	for i := 0; i < len(nn.Weights); i++ {
		outputs = append(outputs, nn.Weights[i].Dot(outputs[i]).ApplyF(nn.Activation))
	}

	return outputs[len(outputs)-1].Data
}

func (nn *NeuralNet) Train(inputsList, targetsList [][]float64, loss func(x, y float64) float64) {
	targets, _ := gnn.NewMatrix(targetsList)
	targets = targets.T()

	inputs, _ := gnn.NewMatrix(inputsList)
	inputs = inputs.T()

	outputs := []*gnn.Matrix{}
	outputs = append(outputs, inputs)
	for i := 0; i < len(nn.Weights); i++ {
		outputs = append(outputs, nn.Weights[i].Dot(outputs[i]).ApplyF(nn.Activation))
	}

	errors := []*gnn.Matrix{}
	errors = append(errors, targets.Elementwise(outputs[len(outputs)-1], loss))
	for i := 0; i != len(nn.Weights); i++ {
		errors = append(errors, nn.Weights[len(nn.Weights)-1-i].T().Dot(errors[i]))
	}

	for i := 0; i < len(nn.Weights); i++ {
		updateWeights := errors[i].ElMult(outputs[len(outputs)-1-i]).ElMult(outputs[len(outputs)-1-i].ApplyF(func(x float64) float64 { return 1.0 - x })).Dot((outputs[len(outputs)-2-i].T())).ScalMult(nn.LR)
		nn.Weights[len(nn.Weights)-1-i] = nn.Weights[len(nn.Weights)-1-i].ElSum(updateWeights)
	}
}

func main() {
	activation := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
	loss := func(x, y float64) float64 { return x - y }

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

	nn, _ := NewNeuralNet([]int{2, 4, 1}, 0.1, activation)

	fmt.Println("XOR NN!")
	fmt.Println("Before trainig:")
	fmt.Println("Resaults:")
	for i := range len(cases) {
		fmt.Println(nn.Query(cases[i]))
	}

	epochs := 10000
	for e := 0; e < epochs; e++ {
		for i := 0; i < len(cases); i++ {
			nn.Train(cases[i], targets[i], loss)
		}
	}
	fmt.Println("Resaults:")
	fmt.Println("After training:")
	for i := range len(cases) {
		fmt.Println(nn.Query(cases[i]))
	}

}
