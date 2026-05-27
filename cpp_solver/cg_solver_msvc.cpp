// cg_solver_msvc.cpp
// Компиляция: cl /O2 /EHsc /LD /openmp /Fe:cg_solver.dll cg_solver_msvc.cpp

#include <vector>
#include <thread>
#include <cmath>
#include <chrono>
#include <mutex>
#include <iostream>

class SparseMatrixCSR {
public:
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    int n;

    SparseMatrixCSR(int size = 0) : n(size) {}

    // Параллельное умножение матрицы на вектор (исправлено: int вместо size_t)
    std::vector<double> matvec_parallel(const std::vector<double>& x, int num_threads) const {
        std::vector<double> result(n, 0.0);

        // Используем OpenMP для параллельного умножения
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

    // Последовательное умножение (для сравнения)
    std::vector<double> matvec_sequential(const std::vector<double>& x) const {
        std::vector<double> result(n, 0.0);

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

// Функция для вычисления скалярного произведения (параллельно)
double dot_product_parallel(const std::vector<double>& a, const std::vector<double>& b, int num_threads) {
    double result = 0.0;

#pragma omp parallel for num_threads(num_threads) reduction(+:result)
    for (int i = 0; i < (int)a.size(); i++) {
        result += a[i] * b[i];
    }

    return result;
}

// Функция для вычисления скалярного произведения (последовательно)
double dot_product_sequential(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    for (int i = 0; i < (int)a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Метод сопряжённых градиентов (параллельная версия)
bool solve_cg_parallel(
    const SparseMatrixCSR& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    int max_iter,
    double tolerance,
    int num_threads,
    int& iterations_done,
    double& final_residual
) {
    int n = A.n;
    x.assign(n, 0.0);

    // r = b - A*x (на первой итерации x=0, значит r=b)
    std::vector<double> r = b;
    std::vector<double> p = r;

    // rsold = r·r
    double rsold = dot_product_parallel(r, r, num_threads);
    double rsold_sqrt = std::sqrt(rsold);

    iterations_done = 0;
    bool converged = false;

    for (int iter = 0; iter < max_iter; iter++) {
        // Ap = A * p (параллельное умножение)
        std::vector<double> Ap = A.matvec_parallel(p, num_threads);

        // pAp = p · Ap (параллельное скалярное произведение)
        double pAp = dot_product_parallel(p, Ap, num_threads);

        if (pAp == 0.0) break;

        // alpha = rsold / pAp
        double alpha = rsold / pAp;

        // x = x + alpha * p
        // r = r - alpha * Ap
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        // rsnew = r · r (параллельное скалярное произведение)
        double rsnew = dot_product_parallel(r, r, num_threads);

        // Проверка сходимости
        if (std::sqrt(rsnew) < tolerance * rsold_sqrt) {
            converged = true;
            iterations_done = iter + 1;
            final_residual = std::sqrt(rsnew);
            break;
        }

        // beta = rsnew / rsold
        double beta = rsnew / rsold;

        // p = r + beta * p (параллельно)
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
        iterations_done = iter + 1;
        final_residual = std::sqrt(rsnew);
    }

    return converged;
}

// Метод сопряжённых градиентов (последовательная версия)
bool solve_cg_sequential(
    const SparseMatrixCSR& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    int max_iter,
    double tolerance,
    int& iterations_done,
    double& final_residual
) {
    int n = A.n;
    x.assign(n, 0.0);

    std::vector<double> r = b;
    std::vector<double> p = r;

    double rsold = dot_product_sequential(r, r);
    double rsold_sqrt = std::sqrt(rsold);

    iterations_done = 0;
    bool converged = false;

    for (int iter = 0; iter < max_iter; iter++) {
        std::vector<double> Ap = A.matvec_sequential(p);

        double pAp = dot_product_sequential(p, Ap);

        if (pAp == 0.0) break;

        double alpha = rsold / pAp;

        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double rsnew = dot_product_sequential(r, r);

        if (std::sqrt(rsnew) < tolerance * rsold_sqrt) {
            converged = true;
            iterations_done = iter + 1;
            final_residual = std::sqrt(rsnew);
            break;
        }

        double beta = rsnew / rsold;

        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
        iterations_done = iter + 1;
        final_residual = std::sqrt(rsnew);
    }

    return converged;
}

extern "C" {

    // Главная функция для вызова из Python
    __declspec(dllexport) double* solve_cg(
        double* A_values, int* A_cols, int* A_row_ptr, int n, int nnz,
        double* b, double tolerance, int max_iter, int num_threads,
        int* iterations, double* residual, int* converged
    ) {
        // Создаём разреженную матрицу
        SparseMatrixCSR A;
        A.n = n;
        A.values.assign(A_values, A_values + nnz);
        A.col_indices.assign(A_cols, A_cols + nnz);
        A.row_ptr.assign(A_row_ptr, A_row_ptr + n + 1);

        // Вектор правой части
        std::vector<double> b_vec(b, b + n);

        // Решение
        std::vector<double> x;
        int iters = 0;
        double resid = 0.0;
        bool conv = false;

        // Измеряем время выполнения
        auto start = std::chrono::high_resolution_clock::now();

        if (num_threads <= 1) {
            conv = solve_cg_sequential(A, b_vec, x, max_iter, tolerance, iters, resid);
        }
        else {
            conv = solve_cg_parallel(A, b_vec, x, max_iter, tolerance, num_threads, iters, resid);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        // Вывод отладочной информации в консоль
        printf("C++ CG Solver: n=%d, threads=%d, iterations=%d, residual=%.2e, time=%.4f sec\n",
            n, num_threads, iters, resid, elapsed);

        *iterations = iters;
        *residual = resid;
        *converged = conv ? 1 : 0;

        // Возвращаем результат
        double* result = new double[n];
        for (int i = 0; i < n; i++) {
            result[i] = x[i];
        }

        return result;
    }

    // Функция для освобождения памяти
    __declspec(dllexport) void free_memory(double* ptr) {
        delete[] ptr;
    }

    __declspec(dllexport) void free_int_memory(int* ptr) {
        delete[] ptr;
    }
}