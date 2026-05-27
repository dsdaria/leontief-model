# Dockerfile для модели Леонтьева с C++ решателем

FROM python:3.10-slim

# Установка компилятора C++ и OpenMP
RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    libomp-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование C++ решателя и компиляция
COPY cpp_solver/ ./cpp_solver/
RUN cd cpp_solver && \
    g++ -shared -fPIC -O3 -fopenmp -o cg_solver.so cg_solver.cpp && \
    cp cg_solver.so .. && \
    ls -la

# Копирование Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование всего проекта
COPY . .

# Открываем порты
EXPOSE 5000 8501

# Запуск удалённого решателя и Streamlit
CMD python remote_solver.py & \
    sleep 3 && \
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0