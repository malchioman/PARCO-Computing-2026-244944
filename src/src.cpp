#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <ctime>
#include <chrono>
using namespace std;


struct coordinate
{
    vector<int> rows;
    vector<int> col;
    vector<float> values;
};

vector<vector<float>> matrixCreation(mt19937& rng, int rows, int cols) {
    uniform_real_distribution prob(0.0f, 1.0f);
    normal_distribution gauss(0.0f, 1000.0f);

    vector A(rows, vector<float>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i][j] = (prob(rng) < 0.5f) ? 0.0f : gauss(rng);
    return A;
}

vector<float> vectorCreation(mt19937& rng, int size) {
    normal_distribution gauss(0.0f, 1000.0f);
    vector<float> v(size);
    for (float& x : v) x = gauss(rng);
    return v;
}

coordinate matrixToCoordinates(vector<vector<float>> matrix)
{
    coordinate coord;
    for (int i = 0; i < matrix.size(); ++i)
    {
        for (int j = 0; j < matrix[i].size(); ++j)
        {
            if (matrix[i][j] == 0.0f){}
            else
            {
                int row = i;
                int col = j;
                float data = matrix[i][j];
                coord.rows.push_back(row);
                coord.col.push_back(col);
                coord.values.push_back(data);

            }
        }
    }
    return coord;
}

vector<float> matrixVectorMoltiplication( coordinate coord, vector<float>& x)
{
    int nrows = 0;
    for (int r : coord.rows)
    {
        if (r + 1 > nrows) nrows = r + 1;
    }
    vector<float> result(nrows, 0.0f);
    for (int i = 0; i < coord.values.size(); ++i)
    {
        result[coord.rows[i]] += coord.values[i] * x[coord.col[i]];
    }
    return result;
}

void printCoordinates(const coordinate& coordinate)
{
    for (int i = 0; i < coordinate.rows.size(); ++i)
    {
        cout << coordinate.rows[i] << ",";
    }
    cout << "\n";
    for (int i = 0; i < coordinate.col.size(); ++i)
    {
        cout << coordinate.col[i] << ",";
    }
    cout << "\n";
    for (int i = 0; i < coordinate.values.size(); ++i)
    {
        cout << coordinate.values[i] << ",";
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

void printResult(const vector<float>& vector)
{
    for (int i = 0; i < vector.size(); ++i)
    {
        cout << vector[i] << "\n";
    }
}

int main() {
    mt19937 rng(static_cast<unsigned>(time(nullptr)));
    uniform_int_distribution<int> distSize(10000,10000);
    int rows = distSize(rng);
    int cols = distSize(rng);

    // data creation
    auto A = matrixCreation(rng, rows, cols);

    //printMatrix(A);
    //cout << "\n";
    //cout << "\n";

    auto x = vectorCreation(rng, cols);
    coordinate coordinate = matrixToCoordinates(A);

    //printCoordinates(coordinate);
    //cout << "\n";
    //cout << "\n";

    auto t0 = chrono::high_resolution_clock::now();
    auto y = matrixVectorMoltiplication(coordinate, x);
    auto t1 = chrono::high_resolution_clock::now();

    //printResult(y);

    auto us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    double ms = us / 1000.0;
    cout << "moltiplication time: " << us << " us ("<< fixed << setprecision(3) << ms << " ms)\n";

    return 0;
}
