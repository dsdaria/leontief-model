#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

extern "C" {

    // Разреженная матрица в формате CSR
    struct SparseMatrixCSR {
        int n;
        double* data;
        int* indices;
        int* indptr;
    };

    // Умножение разреженной матрицы на вектор (параллельное)
    void sparse_matrix_vector_multiply(
        int n,
        double* data,
        int* indices,
        int* indptr,
        double* x,
        double* result,
        int num_threads
    ) {
        omp_set_num_threads(num_threads);

#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            int row_start = indptr[i];
            int row_end = indptr[i + 1];

            for (int j = row_start; j < row_end; j++) {
                sum += data[j] * x[indices[j]];
            }
            result[i] = sum;
        }
    }

    // Ряд Неймана для решения (I-A)X = Y
    void neumann_solver(
        int n,
        double* data,
        int* indices,
        int* indptr,
        double* Y,
        double* X,
        int num_threads,
        int max_iterations
    ) {
        omp_set_num_threads(num_threads);

        // Копируем Y в X и term
        std::vector<double> term(n);
        for (int i = 0; i < n; i++) {
            X[i] = Y[i];
            term[i] = Y[i];
        }

        std::vector<double> new_term(n);

        for (int iter = 1; iter < max_iterations; iter++) {
            // term = A * term
#pragma omp parallel for
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                int row_start = indptr[i];
                int row_end = indptr[i + 1];

                for (int j = row_start; j < row_end; j++) {
                    sum += data[j] * term[indices[j]];
                }
                new_term[i] = sum;
            }

            // Проверка сходимости
            double norm = 0.0;
            for (int i = 0; i < n; i++) {
                norm += new_term[i] * new_term[i];
            }
            norm = std::sqrt(norm);

            if (norm < 1e-8) break;

            // Обновление
#pragma omp parallel for
            for (int i = 0; i < n; i++) {
                X[i] += new_term[i];
                term[i] = new_term[i];
            }
        }
    }

    // Прямой метод Гаусса-Зейделя (параллельный)
    void gauss_seidel(
        int n,
        double* data,
        int* indices,
        int* indptr,
        double* Y,
        double* X,
        int num_threads,
        int max_iterations
    ) {
        omp_set_num_threads(num_threads);

        // Инициализация
        for (int i = 0; i < n; i++) {
            X[i] = 0.0;
        }

        std::vector<double> X_old(n);

        for (int iter = 0; iter < max_iterations; iter++) {
            // Сохраняем старое значение
#pragma omp parallel for
            for (int i = 0; i < n; i++) {
                X_old[i] = X[i];
            }

            // Параллельное обновление (красно-черная схема)
            // Четные индексы
#pragma omp parallel for
            for (int i = 0; i < n; i += 2) {
                double sum = 0.0;
                int row_start = indptr[i];
                int row_end = indptr[i + 1];

                for (int j = row_start; j < row_end; j++) {
                    int col = indices[j];
                    if (col != i) {
                        sum += data[j] * X[col];
                    }
                }
                X[i] = (Y[i] - sum) / data[indptr[i]]; // диагональный элемент
            }

            // Нечетные индексы
#pragma omp parallel for
            for (int i = 1; i < n; i += 2) {
                double sum = 0.0;
                int row_start = indptr[i];
                int row_end = indptr[i + 1];

                for (int j = row_start; j < row_end; j++) {
                    int col = indices[j];
                    if (col != i) {
                        sum += data[j] * X[col];
                    }
                }
                X[i] = (Y[i] - sum) / data[indptr[i]];
            }

            // Проверка сходимости
            double diff = 0.0;
            for (int i = 0; i < n; i++) {
                diff += (X[i] - X_old[i]) * (X[i] - X_old[i]);
            }
            diff = std::sqrt(diff);

            if (diff < 1e-6) break;
        }
    }

} // extern "C"