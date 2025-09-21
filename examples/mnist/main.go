package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/yagnek/gophernn/gnn"
)

func readCSV(name string) ([][]string, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	return records, nil
}

func constructData(loadedCSV [][]string) (inputs, targets [][][]float64, labels []int) {
	inputs = make([][][]float64, 0, len(loadedCSV))
	targets = make([][][]float64, 0, len(loadedCSV))
	labels = make([]int, 0, len(loadedCSV))

	for i := 0; i < len(loadedCSV); i++ {
		dataPiece := make([][]float64, 1)
		dataPiece[0] = make([]float64, len(loadedCSV[i])-1)
		for j := 1; j < len(loadedCSV[i]); j++ {
			p, _ := strconv.ParseFloat(loadedCSV[i][j], 64)
			dataPiece[0][j-1] = p / 255.0
		}
		inputs = append(inputs, dataPiece)

		label, _ := strconv.Atoi(loadedCSV[i][0])
		labels = append(labels, label)

		targetVec := make([]float64, 10)
		targetVec[label] = 1
		targets = append(targets, [][]float64{targetVec})
	}
	return inputs, targets, labels
}

func flattenOutput(out [][]float64) []float64 {
	if len(out) == 0 {
		return nil
	}
	if len(out) == 1 && len(out[0]) > 1 {
		flat := make([]float64, len(out[0]))
		copy(flat, out[0])
		return flat
	}
	flat := make([]float64, len(out))
	for i := 0; i < len(out); i++ {
		if len(out[i]) > 0 {
			flat[i] = out[i][0]
		} else {
			flat[i] = 0.0
		}
	}
	return flat
}

func maxValIdx(vec []float64) int {
	maxIdx := 0
	maxVal := vec[0]
	for i, v := range vec {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

func main() {
	activation := func(x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	}

	trainingCSV, err := readCSV("./data/mnist_train.csv")
	if err != nil {
		log.Fatal(err)
	}

	testCSV, err := readCSV("./data/mnist_test.csv")
	if err != nil {
		log.Fatal(err)
	}

	trainingInputs, trainingTargets, _ := constructData(trainingCSV)
	testInputs, _, testLabels := constructData(testCSV)

	nn := gnn.NewNeuralNet([]int{784, 200, 10}, 0.1, activation)

	epochs := 10
	for e := range epochs {
		fmt.Println("Epoch", e+1)
		for i := 0; i < 1000; i++ {
			nn.Train(trainingInputs[i], trainingTargets[i])
		}
	}

	correct := 0
	total := 100

	for i := 0; i < total; i++ {
		out2d := nn.Query(testInputs[i])
		out := flattenOutput(out2d)
		guess := maxValIdx(out)
		answer := testLabels[i]

		if guess == answer {
			correct++
		}
	}
	accuracy := float64(correct) / float64(total) * 100.0
	fmt.Printf("\nClassification accuracy on %d test samples: %.2f%%\n", total, accuracy)

}
