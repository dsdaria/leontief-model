#ifndef CG_SOLVER_H
#define CG_SOLVER_H

#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>
#include <iostream>

// Разреженная матрица в формате CSR
class SparseMatrixCSR {
public:
    std::vector<double> values;      // Значения ненулевых элементов
    std::vector<int> col_indices;    // Индексы столбцов
    std::vector<int> row_ptr;        // Указатели на начало строк
    int n;                            // Размер матрицы

    SparseMatrixCSR(int size = 0) : n(size) {}

    // Заполнение из плотной матрицы
    void from_dense(const std::vector<double>& dense, int size);

    // Умножение матрицы на вектор (A * x)
    std::vector<double> matvec(const std::vector<double>& x) const;

    // Параллельное умножение матрицы на вектор
    std::vector<double> matvec_parallel(const std::vector<double>& x, int num_threads) const;
};

// Параллельный решатель CG метода
class ParallelCGSolver {
private:
    int n;
    double tolerance;
    int max_iterations;
    int num_threads;

public:
    ParallelCGSolver(int matrix_size = 0, double tol = 1e-8, int max_iter = 1000, int threads = 4);

    // Решение СЛАУ A * x = b
    struct Solution {
        std::vector<double> x;
        int iterations;
        double residual;
        double computation_time;
        bool converged;
    };

    Solution solve(const SparseMatrixCSR& A, const std::vector<double>& b);

    // Параллельное решение для нескольких правых частей
    std::vector<Solution> solve_batch(const SparseMatrixCSR& A, const std::vector<std::vector<double>>& B);
};

// Функции для экспорта в Python
extern "C" {
    // Решение одной системы
    double* solve_cg(double* A_values, int* A_cols, int* A_row_ptr, int n, int nnz,
        double* b, double tolerance, int max_iter, int num_threads,
        int* iterations, double* residual, int* converged);

    // Решение нескольких систем параллельно
    double* solve_batch_cg(double* A_values, int* A_cols, int* A_row_ptr, int n, int nnz,
        double* b_matrix, int n_rhs, double tolerance,
        int max_iter, int num_threads);

    // Генерация случайной разреженной матрицы
    void generate_sparse_matrix(int n, double density, int seed,
        double** values, int** cols, int** row_ptr, int* nnz);

    // Освобождение памяти
    void free_memory(double* ptr);
    void free_int_memory(int* ptr);
}

#endif // CG_SOLVER_H