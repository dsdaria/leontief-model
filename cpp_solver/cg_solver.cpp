// cg_solver.cpp
// Решение СЛАУ методом сопряжённых градиентов (CG)
// Кроссплатформенная версия (Windows/Linux)

#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <omp.h>

// Кроссплатформенный экспорт символов
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// Класс разреженной матрицы в формате CSR
class SparseMatrixCSR {
public:
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    int n;

    SparseMatrixCSR(int size = 0) : n(size) {}

    // Параллельное умножение матрицы на вектор
    std::vector<double> matvec(const std::vector<double>& x, int num_threads) const {
        std::vector<double> result(n, 0.0);
#pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                sum += values[j] * x[col_indices[j]];
            }
            result[i] = sum;
        }
        return result;
    }

    // Параллельное скалярное произведение
    double dot_product(const std::vector<double>& a, const std::vector<double>& b, int num_threads) {
        double result = 0.0;
#pragma omp parallel for num_threads(num_threads) reduction(+:result)
        for (int i = 0; i < (int)a.size(); i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    // Метод сопряжённых градиентов
    int conjugate_gradient(
        const SparseMatrixCSR& A,
        const std::vector<double>& b,
        std::vector<double>& x,
        int max_iter,
        double tolerance,
        int num_threads,
        double& residual_norm
    ) {
        int n = A.n;
        x.assign(n, 0.0);

        std::vector<double> r = b;
        std::vector<double> p = r;

        double rsold = dot_product(r, r, num_threads);
        double rsold_sqrt = std::sqrt(rsold);

        for (int iter = 0; iter < max_iter; iter++) {
            std::vector<double> Ap = A.matvec(p, num_threads);
            double pAp = dot_product(p, Ap, num_threads);

            if (pAp == 0.0) break;

            double alpha = rsold / pAp;

#pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < n; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            double rsnew = dot_product(r, r, num_threads);
            double rsnew_sqrt = std::sqrt(rsnew);

            if (rsnew_sqrt < tolerance * rsold_sqrt) {
                residual_norm = rsnew_sqrt;
                return iter + 1;
            }

            double beta = rsnew / rsold;

#pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < n; i++) {
                p[i] = r[i] + beta * p[i];
            }

            rsold = rsnew;
            residual_norm = rsnew_sqrt;
        }

        return max_iter;
    }
};

// ============================================================
// ФУНКЦИИ ДЛЯ ЭКСПОРТА В PYTHON
// ============================================================

extern "C" {

    // Решение СЛАУ для одной правой части
    EXPORT double* solve_cg(
        double* values, int* col_indices, int* row_ptr,
        int n, int nnz, double* b,
        double tolerance, int max_iter, int num_threads,
        int* iterations, double* residual, int* converged
    ) {
        // Создаём разреженную матрицу
        SparseMatrixCSR A;
        A.n = n;
        A.values.assign(values, values + nnz);
        A.col_indices.assign(col_indices, col_indices + nnz);
        A.row_ptr.assign(row_ptr, row_ptr + n + 1);

        std::vector<double> b_vec(b, b + n);
        std::vector<double> x;
        double resid = 0.0;

        auto start = std::chrono::high_resolution_clock::now();
        int iters = A.conjugate_gradient(A, b_vec, x, max_iter, tolerance, num_threads, resid);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        printf("CG: n=%d, thr=%d, iter=%d, resid=%.2e, time=%.4f\n", n, num_threads, iters, resid, elapsed);

        *iterations = iters;
        *residual = resid;
        *converged = (resid < tolerance) ? 1 : 0;

        double* result = new double[n];
        for (int i = 0; i < n; i++) {
            result[i] = x[i];
        }
        return result;
    }

    // Решение СЛАУ для множества правых частей (многовариантные расчёты)
    EXPORT double* solve_batch_cg(
        double* values, int* col_indices, int* row_ptr,
        int n, int nnz, double* b_matrix, int n_rhs,
        double tolerance, int max_iter, int num_threads
    ) {
        printf("\n========== BATCH CG ==========\n");
        printf("n=%d, n_rhs=%d, threads=%d\n", n, n_rhs, num_threads);

        // Создаём разреженную матрицу
        SparseMatrixCSR A;
        A.n = n;
        A.values.assign(values, values + nnz);
        A.col_indices.assign(col_indices, col_indices + nnz);
        A.row_ptr.assign(row_ptr, row_ptr + n + 1);

        // Формируем правые части
        std::vector<std::vector<double>> B(n_rhs, std::vector<double>(n));
        for (int k = 0; k < n_rhs; k++) {
            for (int i = 0; i < n; i++) {
                B[k][i] = b_matrix[k * n + i];
            }
        }

        std::vector<std::vector<double>> results(n_rhs, std::vector<double>(n, 0.0));

        auto start = std::chrono::high_resolution_clock::now();

        // Параллельное решение для каждого сценария
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int k = 0; k < n_rhs; k++) {
            std::vector<double> x;
            double resid = 0.0;
            A.conjugate_gradient(A, B[k], x, max_iter, tolerance, 1, resid);
            for (int i = 0; i < n; i++) {
                results[k][i] = x[i];
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        printf("Batch time: %.4f sec (%.4f per system)\n", elapsed, elapsed / n_rhs);
        printf("===============================\n");

        double* all_results = new double[n_rhs * n];
        for (int k = 0; k < n_rhs; k++) {
            for (int i = 0; i < n; i++) {
                all_results[k * n + i] = results[k][i];
            }
        }
        return all_results;
    }

    // Освобождение памяти
    EXPORT void free_memory(double* ptr) {
        delete[] ptr;
    }

} // extern "C"