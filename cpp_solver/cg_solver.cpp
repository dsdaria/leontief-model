// cg_solver_full.cpp
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <omp.h>

class SparseMatrixCSR {
public:
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    int n;

    SparseMatrixCSR(int size = 0) : n(size) {}

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
};

double dot_product(const std::vector<double>& a, const std::vector<double>& b, int num_threads) {
    double result = 0.0;
#pragma omp parallel for num_threads(num_threads) reduction(+:result)
    for (int i = 0; i < (int)a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

int conjugate_gradient(
    const SparseMatrixCSR& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    int max_iter, double tolerance, int num_threads, double& residual_norm
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
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
        rsold = rsnew;
        residual_norm = rsnew_sqrt;
    }
    return max_iter;
}

extern "C" {

    __declspec(dllexport) double* solve_cg(
        double* A_values, int* A_cols, int* A_row_ptr, int n, int nnz,
        double* b, double tolerance, int max_iter, int num_threads,
        int* iterations, double* residual, int* converged
    ) {
        SparseMatrixCSR A;
        A.n = n;
        A.values.assign(A_values, A_values + nnz);
        A.col_indices.assign(A_cols, A_cols + nnz);
        A.row_ptr.assign(A_row_ptr, A_row_ptr + n + 1);

        std::vector<double> b_vec(b, b + n);
        std::vector<double> x;
        double resid = 0.0;

        auto start = std::chrono::high_resolution_clock::now();
        int iters = conjugate_gradient(A, b_vec, x, max_iter, tolerance, num_threads, resid);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        printf("CG: n=%d, thr=%d, iter=%d, resid=%.2e, time=%.4f\n", n, num_threads, iters, resid, elapsed);

        *iterations = iters;
        *residual = resid;
        *converged = (resid < tolerance) ? 1 : 0;

        double* result = new double[n];
        for (int i = 0; i < n; i++) result[i] = x[i];
        return result;
    }

    __declspec(dllexport) double* solve_batch_cg(
        double* A_values, int* A_cols, int* A_row_ptr, int n, int nnz,
        double* b_matrix, int n_rhs, double tolerance, int max_iter, int num_threads
    ) {
        printf("\n========== BATCH CG ==========\n");
        printf("n=%d, n_rhs=%d, threads=%d\n", n, n_rhs, num_threads);

        SparseMatrixCSR A;
        A.n = n;
        A.values.assign(A_values, A_values + nnz);
        A.col_indices.assign(A_cols, A_cols + nnz);
        A.row_ptr.assign(A_row_ptr, A_row_ptr + n + 1);

        std::vector<std::vector<double>> B(n_rhs, std::vector<double>(n));
        for (int k = 0; k < n_rhs; k++) {
            for (int i = 0; i < n; i++) {
                B[k][i] = b_matrix[k * n + i];
            }
        }

        std::vector<std::vector<double>> results(n_rhs, std::vector<double>(n, 0.0));

        auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int k = 0; k < n_rhs; k++) {
            std::vector<double> x;
            double resid = 0.0;
            conjugate_gradient(A, B[k], x, max_iter, tolerance, 1, resid);
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

    __declspec(dllexport) void free_memory(double* ptr) {
        delete[] ptr;
    }
}