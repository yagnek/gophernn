## GopherNN

A small Go deep learning library, written for learning purposes.

### Features
- Define neural networks with an arbitrary number of layers
- Plug in your own activation function
- Train networks with backpropagation
- Query networks for predictions
- No dependencies outside the Go standard library


### API
```
func NewNeuralNet(shape []int, lr float64, activation func(x float64) float64) *NeuralNet
func (nn *NeuralNet) Query(inputsList [][]float64) [][]float64
func (nn *NeuralNet) Train(inputsList, targetsList [][]float64)

```

### Examples
See the [examples/](./examples) folder for XOR and MNIST runnable demos.
