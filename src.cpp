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

float vector_avarage(vector<float> v){
    float total = 0.0f;
    for (auto i = 0u; i < v.size(); i++){
        total += v[i];
    }
    return total / v.size();
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

    /*
    //====================================================================================
    //FIRST EXPERIMENT -------- SAME MATRIX 50K x 50K BUT WITH DIFFERENT NUMBER OF THREADS
    //====================================================================================


    vector<float> time1;
    vector<float> time2;
    vector<float> time3;
    vector<float> time4;
    vector<float> time5;
    vector<float> time6;
    vector<float> time7;

    for (int i = 0; i <= 10000; i++) {
        int p1 = 1;
        auto t0 = chrono::high_resolution_clock::now();
        auto y1 = matrixVectorMoltiplication(coordinate, x, p1);
        auto t1 = chrono::high_resolution_clock::now();

        auto us1 = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
        double ms1 = us1 / 1000.0;
        time1.push_back(ms1);

        int p2 = 2;
        auto t2 = chrono::high_resolution_clock::now();
        auto y2 = matrixVectorMoltiplication(coordinate, x, p2);
        auto t3 = chrono::high_resolution_clock::now();

        auto us2 = chrono::duration_cast<chrono::microseconds>(t3 - t2).count();
        double ms2 = us2 / 1000.0;
        time2.push_back(ms2);

        int p3 = 4;
        auto t4 = chrono::high_resolution_clock::now();
        auto y3 = matrixVectorMoltiplication(coordinate, x, p3);
        auto t5 = chrono::high_resolution_clock::now();

        auto us3 = chrono::duration_cast<chrono::microseconds>(t5 - t4).count();
        double ms3 = us3 / 1000.0;
        time3.push_back(ms3);

        int p4 = 8;
        auto t6 = chrono::high_resolution_clock::now();
        auto y4 = matrixVectorMoltiplication(coordinate, x, p4);
        auto t7 = chrono::high_resolution_clock::now();

        auto us4 = chrono::duration_cast<chrono::microseconds>(t7 - t6).count();
        double ms4 = us4 / 1000.0;
        time4.push_back(ms4);

        int p5 = 12;
        auto t8 = chrono::high_resolution_clock::now();
        auto y5 = matrixVectorMoltiplication(coordinate, x, p5);
        auto t9 = chrono::high_resolution_clock::now();

        auto us5 = chrono::duration_cast<chrono::microseconds>(t9 - t8).count();
        double ms5 = us5 / 1000.0;
        time5.push_back(ms5);

        int p6 = 16;
        auto t10 = chrono::high_resolution_clock::now();
        auto y6 = matrixVectorMoltiplication(coordinate, x, p6);
        auto t11 = chrono::high_resolution_clock::now();

        auto us6 = chrono::duration_cast<chrono::microseconds>(t11 - t10).count();
        double ms6 = us6 / 1000.0;
        time6.push_back(ms6);

        int p7 = 20;
        auto t12 = chrono::high_resolution_clock::now();
        auto y7 = matrixVectorMoltiplication(coordinate, x, p7);
        auto t13 = chrono::high_resolution_clock::now();

        auto us7 = chrono::duration_cast<chrono::microseconds>(t13 - t12).count();
        double ms7 = us7 / 1000.0;
        time7.push_back(ms7);
    }

    float thread1 = vector_avarage(time1);
    float thread2 = vector_avarage(time2);
    float thread4 = vector_avarage(time3);
    float thread8 = vector_avarage(time4);
    float thread12 = vector_avarage(time5);
    float thread16 = vector_avarage(time6);
    float thread20 = vector_avarage(time7);

    cout << thread1 << endl;
    cout << thread2 << endl;
    cout << thread4 << endl;
    cout << thread8 << endl;
    cout << thread12 << endl;
    cout << thread16 << endl;
    cout << thread20 << endl;
    */


    //=================================================================================================
    //SECOND EXPERIMENT -------- DIFFERENT MATRIX SIZE, COMPARISON WITH 1 AND 20(max) NUMBER OF THREADS
    //=================================================================================================

    vector<float> time1;
    vector<float> time7;

    for (int i = 0; i < 1000; i++) {

        int p1 = 1;
        auto t0 = chrono::high_resolution_clock::now();
        auto y1 = matrixVectorMoltiplication(coordinate, x, p1);
        auto t1 = chrono::high_resolution_clock::now();

        auto us1 = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
        double ms1 = us1 / 1000.0;
        time1.push_back(ms1);

        int p7 = 20;
        auto t12 = chrono::high_resolution_clock::now();
        auto y7 = matrixVectorMoltiplication(coordinate, x, p7);
        auto t13 = chrono::high_resolution_clock::now();

        auto us7 = chrono::duration_cast<chrono::microseconds>(t13 - t12).count();
        double ms7 = us7 / 1000.0;
        time7.push_back(ms7);
    }

    float thread1 = vector_avarage(time1);
    float thread7 = vector_avarage(time7);

    cout << thread1 << endl;
    cout << thread7 << endl;

    return 0;
}