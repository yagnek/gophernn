package gnn

import (
	"fmt"
	"math"
	"math/rand"
)

type matrix struct {
	Rows int
	Cols int
	Data [][]float64
}

func newMatrix(data [][]float64) (m *matrix, err error) {
	rows := len(data)
	cols := len(data[0])
	for i := range rows {
		if len(data[i]) != cols {
			err := fmt.Errorf("cannot construct matrix: the amount columns in data is not uniform")
			return nil, err
		}
	}
	return &matrix{
		Rows: len(data),
		Cols: len(data[0]),
		Data: data,
	}, nil
}

func zeroMatrix(rows, cols int) *matrix {
	mData := make([][]float64, rows)
	for i := range mData {
		mData[i] = make([]float64, cols)
	}
	return &matrix{
		Rows: rows,
		Cols: cols,
		Data: mData,
	}
}

func (m *matrix) initRandom() *matrix {
	fanIn := m.Cols
	scale := 1.0 / math.Sqrt(float64(fanIn))
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] = rand.NormFloat64() * scale
		}
	}
	return m
}

func (m *matrix) show() {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			fmt.Print(m.Data[i][j], " ")
		}
		fmt.Println()
	}
	fmt.Println()
}

func (m *matrix) T() *matrix {
	out := zeroMatrix(m.Cols, m.Rows)
	for i := range m.Rows {
		for j := range m.Cols {
			out.Data[j][i] = m.Data[i][j]
		}
	}
	return out
}

func (m *matrix) applyF(f func(x float64) float64) *matrix {
	out := zeroMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			out.Data[i][j] = f(m.Data[i][j])
		}
	}
	return out
}

func (m *matrix) scalMult(s float64) *matrix {
	return m.applyF(func(x float64) float64 { return x * s })
}

func (a *matrix) elementwise(b *matrix, f func(x, y float64) float64) *matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("cannot perform elementwise operation: dimensions %dx%d and %dx%d", a.Rows, a.Cols, b.Rows, b.Cols))
	}
	m := zeroMatrix(a.Rows, a.Cols)
	for i := range a.Rows {
		for j := range b.Cols {
			m.Data[i][j] = f(a.Data[i][j], b.Data[i][j])
		}
	}
	return m
}

func (a *matrix) elMult(b *matrix) *matrix {
	return a.elementwise(b, func(x float64, y float64) float64 { return x * y })
}

func (a *matrix) elDiv(b *matrix) *matrix {
	return a.elementwise(b, func(x float64, y float64) float64 { return x / y })
}

func (a *matrix) elSum(b *matrix) *matrix {
	return a.elementwise(b, func(x float64, y float64) float64 { return x + y })
}

func (a *matrix) elSub(b *matrix) *matrix {
	return a.elementwise(b, func(x float64, y float64) float64 { return x - y })
}

func (a *matrix) dot(b *matrix) (m *matrix) {
	if a.Cols != b.Rows {
		panic(fmt.Sprintf("cannot multiply: dimensions %dx%d and %dx%d", a.Rows, a.Cols, b.Rows, b.Cols))
	}
	m = zeroMatrix(a.Rows, b.Cols)
	for i := range m.Rows {
		for j := range m.Cols {
			for n := range a.Cols {
				m.Data[i][j] += a.Data[i][n] * b.Data[n][j]
			}
		}
	}
	return m
}
