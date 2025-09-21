package gnn

import (
	"fmt"
)

type Matrix struct {
	Rows int
	Cols int
	Data [][]float64
}

func NewMatrix(data [][]float64) (m *Matrix, err error) {
	rows := len(data)
	cols := len(data[0])
	for i := range rows {
		if len(data[i]) != cols {
			err := fmt.Errorf("cannot construct matrix: the amount columns in data is not uniform")
			return nil, err
		}
	}
	return &Matrix{
		Rows: len(data),
		Cols: len(data[0]),
		Data: data,
	}, nil
}

func ZeroMatrix(rows, cols int) *Matrix {
	mData := make([][]float64, rows)
	for i := range mData {
		mData[i] = make([]float64, cols)
	}
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: mData,
	}
}

func (m *Matrix) Show() {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			fmt.Print(m.Data[i][j], " ")
		}
		fmt.Println()
	}
	fmt.Println()
}

func (m *Matrix) T() *Matrix {
	out := ZeroMatrix(m.Cols, m.Rows)
	for i := range m.Rows {
		for j := range m.Cols {
			out.Data[j][i] = m.Data[i][j]
		}
	}
	return out
}

func (m *Matrix) ApplyF(f func(x float64) float64) *Matrix {
	out := ZeroMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			out.Data[i][j] = f(m.Data[i][j])
		}
	}
	return out
}

func (m *Matrix) ScalMult(s float64) *Matrix {
	return m.ApplyF(func(x float64) float64 { return x * s })
}

func (a *Matrix) Elementwise(b *Matrix, f func(x, y float64) float64) *Matrix {
	m := ZeroMatrix(a.Rows, a.Cols)
	for i := range a.Rows {
		for j := range b.Cols {
			m.Data[i][j] = f(a.Data[i][j], b.Data[i][j])
		}
	}
	return m
}

func (a *Matrix) ElMult(b *Matrix) *Matrix {
	return a.Elementwise(b, func(x float64, y float64) float64 { return x * y })
}

func (a *Matrix) ElDiv(b *Matrix) *Matrix {
	return a.Elementwise(b, func(x float64, y float64) float64 { return x / y })
}

func (a *Matrix) ElSum(b *Matrix) *Matrix {
	return a.Elementwise(b, func(x float64, y float64) float64 { return x + y })
}

func (a *Matrix) ElSub(b *Matrix) *Matrix {
	return a.Elementwise(b, func(x float64, y float64) float64 { return x - y })
}

func (a *Matrix) Dot(b *Matrix) (m *Matrix) {
	if a.Cols != b.Rows {
		panic(fmt.Sprintf("cannot multiply: dimensions %dx%d and %dx%d", a.Rows, a.Cols, b.Rows, b.Cols))
	}
	m = ZeroMatrix(a.Rows, b.Cols)
	for i := range m.Rows {
		for j := range m.Cols {
			for n := range a.Cols {
				m.Data[i][j] += a.Data[i][n] * b.Data[n][j]
			}
		}
	}
	return m
}
