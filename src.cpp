#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <omp.h>
using namespace std;


struct coordinate
{
    int rows, cols;
    std::vector<int>   rowptr; // size = rows+1
    std::vector<int>   col;    // size = nnz
    std::vector<float> val;    // size = nnz
};

vector<vector<float>> matrixCreation(mt19937& rng, int rows, int cols) {
    uniform_real_distribution prob(0.0f, 1.0f);
    normal_distribution gauss(0.0f, 10.0f);

    vector A(rows, vector<float>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i][j] = (prob(rng) < 0.5f) ? 0.0f : gauss(rng);
    return A;
}

vector<float> vectorCreation(mt19937& rng, int size) {
    normal_distribution gauss(0.0f, 10.0f);
    vector<float> v(size);
    for (float& x : v) x = gauss(rng);
    return v;
}

coordinate matrixToCoordinates(vector<vector<float>> A)
{
    coordinate csr;
    csr.rows = A.size();
    csr.cols = csr.rows ? A[0].size() : 0;
    csr.rowptr.resize(csr.rows + 1, 0);

    // Primo passaggio: conta quanti elementi non nulli per riga
    for (int i = 0; i < csr.rows; ++i) {
        for (int j = 0; j < csr.cols; ++j) {
            if (A[i][j] != 0.0f)
                csr.rowptr[i + 1]++;
        }
    }

    // Prefix sum → rowptr[i] diventa l'indice di inizio di ogni riga
    for (int i = 0; i < csr.rows; ++i)
        csr.rowptr[i + 1] += csr.rowptr[i];

    // Alloca spazio per i dati
    int nnz = csr.rowptr.back();
    csr.col.resize(nnz);
    csr.val.resize(nnz);

    // Secondo passaggio: riempi col e val
    std::vector<int> next = csr.rowptr; // cursori per ogni riga
    for (int i = 0; i < csr.rows; ++i) {
        for (int j = 0; j < csr.cols; ++j) {
            float v = A[i][j];
            if (v != 0.0f) {
                int p = next[i]++;
                csr.col[p] = j;
                csr.val[p] = v;
            }
        }
    }

    return csr;
}

vector<float> matrixVectorMoltiplication(const coordinate& A,
                                         const vector<float>& x, int p)
{
    vector<float> y(A.rows, 0.0f);

#pragma omp parallel num_threads(p) default(none) shared(A,x,y)
    {
#pragma omp for schedule(static,64)
        for (int i = 0; i < A.rows; ++i) {
            float acc = 0.0f;                 // privato al thread
            for (int k = A.rowptr[i]; k < A.rowptr[i+1]; ++k)
                acc += A.val[k] * x[A.col[k]];
            y[i] = acc;                       // ogni iter scrive un indice diverso
        }
    }
    return y;
}

/*
void printCoordinates(const coordinate& coordinate)
{

    for (int i = 0; i < coordinate.rowptr.size(); ++i)
    {
        cout << coordinate.rowptr[i] << ",";
    }
    cout << "\n";
    for (int i = 0; i < coordinate.col.size(); ++i)
    {
        cout << coordinate.col[i] << ",";
    }
    cout << "\n";
    for (int i = 0; i < coordinate.val.size(); ++i)
    {
        cout << coordinate.val[i] << ",";
    }

}

void printMatrix(const vector<vector<float>>& vector)
{
    for (int i = 0; i < vector.size(); ++i)
    {
        for (int j = 0; j < vector[i].size(); ++j)
        {
            cout << vector[i][j] << ",";
        }
        cout << "\n";
    }
};

void printVector(const vector<float>& vector)
{
    for (int i = 0; i < vector.size(); ++i)
    {
        cout << vector[i] << "\n";
    }
}
*/

int main() {
    mt19937 rng(static_cast<unsigned>(time(nullptr)));
    uniform_int_distribution<int> distSize(1000,1000);
    int rows;
    cout << "Please enter the number of rows of the matrix" << endl;
    cin >> rows;
    int cols;
    cout << "Please enter the number of columns of the matrix" << endl;
    cin >> cols;
    auto A = matrixCreation(rng, rows, cols);
    auto x = vectorCreation(rng, cols);
    coordinate coordinate = matrixToCoordinates(A);
    int n = omp_get_max_threads();
    cout << "insert the number of threads that you wanna use, the max number of avaliable threads is: " << n << endl;
    bool w = true;
    int p;
    while (w)
    {
        cin >> p;
        if (p<=n){
            w = false;
        }else
        {
            cout << "insert a number that is smaller than " << n << endl;
        }
    }
    omp_set_dynamic(0);



    auto t0 = chrono::high_resolution_clock::now();
    auto y = matrixVectorMoltiplication(coordinate, x, p);
    auto t1 = chrono::high_resolution_clock::now();
    //printMatrix(A);
    //cout << "\n";
    //cout << "\n";
    //printVector(x);
    //cout << "\n";
    //cout << "\n";
    //printCoordinates(coordinate);
    //cout << "\n";
    //cout << "\n";
    //printVector(y);

    auto us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    double ms = us / 1000.0;
    cout << "moltiplication time: " << us << " us ("<< fixed << setprecision(3) << ms << " ms)\n";

    return 0;
}