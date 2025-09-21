package gnn

import (
	"math"
	"math/rand"
)

type NeuralNet struct {
	Shape      []int
	LR         float64
	Activation func(x float64) float64
	Weights    []*Matrix
}

func InitRandom(m *Matrix) *Matrix {
	fanIn := m.Cols
	scale := 1.0 / math.Sqrt(float64(fanIn))
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] = rand.NormFloat64() * scale
		}
	}
	return m
}
func NewNeuralNet(shape []int, lr float64, activation func(x float64) float64) (nn *NeuralNet, err error) {
	weights := []*Matrix{}
	for i := range (len(shape)) - 1 {
		weights = append(weights, InitRandom(ZeroMatrix(shape[i+1], shape[i])))
	}
	return &NeuralNet{
		Shape:      shape,
		LR:         lr,
		Activation: activation,
		Weights:    weights,
	}, nil
}

func (nn *NeuralNet) Query(inputsList [][]float64) [][]float64 {
	inputs, _ := NewMatrix(inputsList)
	inputs = inputs.T()

	outputs := []*Matrix{}
	outputs = append(outputs, inputs)
	for i := 0; i < len(nn.Weights); i++ {
		outputs = append(outputs, nn.Weights[i].Dot(outputs[i]).ApplyF(nn.Activation))
	}

	return outputs[len(outputs)-1].Data
}

func (nn *NeuralNet) Train(inputsList, targetsList [][]float64) {
	targets, _ := NewMatrix(targetsList)
	targets = targets.T()

	inputs, _ := NewMatrix(inputsList)
	inputs = inputs.T()

	outputs := []*Matrix{}
	outputs = append(outputs, inputs)
	for i := 0; i < len(nn.Weights); i++ {
		outputs = append(outputs, nn.Weights[i].Dot(outputs[i]).ApplyF(nn.Activation))
	}

	lossDerivativeFn := func(x, y float64) float64 { return x - y }

	errors := []*Matrix{}
	errors = append(errors, targets.Elementwise(outputs[len(outputs)-1], lossDerivativeFn))
	for i := 0; i < len(nn.Weights); i++ {
		errors = append(errors, nn.Weights[len(nn.Weights)-1-i].T().Dot(errors[i]))
	}

	for i := 0; i < len(nn.Weights); i++ {
		updateWeights := errors[i].ElMult(outputs[len(outputs)-1-i]).ElMult(outputs[len(outputs)-1-i].ApplyF(func(x float64) float64 { return 1.0 - x })).Dot((outputs[len(outputs)-2-i].T())).ScalMult(nn.LR)
		nn.Weights[len(nn.Weights)-1-i] = nn.Weights[len(nn.Weights)-1-i].ElSum(updateWeights)
	}
}
