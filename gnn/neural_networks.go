package gnn

import (
	"log"
	"math"
	"math/rand"
)

type NeuralNet struct {
	Shape      []int
	LR         float64
	Activation func(x float64) float64
	Weights    []*matrix
}

func InitRandom(m *matrix) *matrix {
	fanIn := m.Cols
	scale := 1.0 / math.Sqrt(float64(fanIn))
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] = rand.NormFloat64() * scale
		}
	}
	return m
}
func NewNeuralNet(shape []int, lr float64, activation func(x float64) float64) *NeuralNet {
	weights := []*matrix{}
	for i := range (len(shape)) - 1 {
		weights = append(weights, InitRandom(zeroMatrix(shape[i+1], shape[i])))
	}
	return &NeuralNet{
		Shape:      shape,
		LR:         lr,
		Activation: activation,
		Weights:    weights,
	}
}

func (nn *NeuralNet) Query(inputsList [][]float64) [][]float64 {
	inputs, err := newMatrix(inputsList)
	if err != nil {
		log.Fatal(err)
	}
	inputs = inputs.T()

	outputs := []*matrix{}
	outputs = append(outputs, inputs)
	for i := 0; i < len(nn.Weights); i++ {
		outputs = append(outputs, nn.Weights[i].dot(outputs[i]).applyF(nn.Activation))
	}

	return outputs[len(outputs)-1].Data
}

func (nn *NeuralNet) Train(inputsList, targetsList [][]float64) {
	targets, err := newMatrix(targetsList)
	if err != nil {
		log.Fatal(err)
	}

	targets = targets.T()

	inputs, nil := newMatrix(inputsList)
	if err != nil {
		log.Fatal(err)
	}

	inputs = inputs.T()

	outputs := []*matrix{}
	outputs = append(outputs, inputs)

	for i := 0; i < len(nn.Weights); i++ {
		outputs = append(outputs, nn.Weights[i].dot(outputs[i]).applyF(nn.Activation))
	}

	lossDerivativeFn := func(x, y float64) float64 { return x - y }

	errors := []*matrix{}
	errors = append(errors, targets.elementwise(outputs[len(outputs)-1], lossDerivativeFn))
	for i := 0; i < len(nn.Weights); i++ {
		errors = append(errors, nn.Weights[len(nn.Weights)-1-i].T().dot(errors[i]))
	}

	for i := 0; i < len(nn.Weights); i++ {
		updateWeights := errors[i].elMult(outputs[len(outputs)-1-i]).elMult(outputs[len(outputs)-1-i].applyF(func(x float64) float64 { return 1.0 - x })).dot((outputs[len(outputs)-2-i].T())).scalMult(nn.LR)
		nn.Weights[len(nn.Weights)-1-i] = nn.Weights[len(nn.Weights)-1-i].elSum(updateWeights)
	}
}
