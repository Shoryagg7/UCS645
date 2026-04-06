#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <omp.h>
#include <iomanip>
#include <algorithm>

using namespace std;

const int MATCH = 2;
const int MISMATCH = -1;
const int GAP = -1;

// Smith-Waterman local sequence alignment
class SmithWaterman {
public:
    string seq1, seq2;
    vector<vector<int>> matrix;
    int rows, cols;

    SmithWaterman(const string& s1, const string& s2) : seq1(s1), seq2(s2) {
        rows = seq1.length() + 1;
        cols = seq2.length() + 1;
        matrix.assign(rows, vector<int>(cols, 0));
    }

    int getScore(char a, char b) {
        return (a == b) ? MATCH : MISMATCH;
    }

    // Serial implementation
    void computeSerial() {
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < cols; j++) {
                int match = matrix[i-1][j-1] + getScore(seq1[i-1], seq2[j-1]);
                int delete_op = matrix[i-1][j] + GAP;
                int insert_op = matrix[i][j-1] + GAP;

                matrix[i][j] = max({0, match, delete_op, insert_op});
            }
        }
    }

    // Parallel wavefront implementation
    void computeParallel(int num_threads) {
        for (int diagonal = 2; diagonal < rows + cols; diagonal++) {
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
            for (int i = 1; i < rows; i++) {
                int j = diagonal - i;
                if (j >= 1 && j < cols) {
                    int match = matrix[i-1][j-1] + getScore(seq1[i-1], seq2[j-1]);
                    int delete_op = matrix[i-1][j] + GAP;
                    int insert_op = matrix[i][j-1] + GAP;

                    matrix[i][j] = max({0, match, delete_op, insert_op});
                }
            }
        }
    }

    int getMaxScore() {
        int max_score = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                max_score = max(max_score, matrix[i][j]);
            }
        }
        return max_score;
    }
};

// Generate random DNA sequence
string generateDNASequence(int length) {
    string seq = "";
    char bases[] = {'A', 'T', 'G', 'C'};
    for (int i = 0; i < length; i++) {
        seq += bases[rand() % 4];
    }
    return seq;
}

int main() {
    int seq_length = 2000;  // Length of sequences

    string seq1 = generateDNASequence(seq_length);
    string seq2 = generateDNASequence(seq_length);

    ofstream results("smithwaterman_results.txt");
    results << "#Threads\tTime(s)\tSpeedup\tMax_Score\n";

    double serial_time = 0.0;
    int baseline_score = 0;

    cout << "\nSmith-Waterman Parallel DNA Sequence Alignment\n";
    cout << "=" << string(59, '=') << "\n";
    cout << "Sequence 1 length: " << seq1.length() << "\n";
    cout << "Sequence 2 length: " << seq2.length() << "\n";
    cout << "Matrix size: " << (seq1.length() + 1) << " x " << (seq2.length() + 1) << "\n\n";

    for (int num_threads = 1; num_threads <= 16; num_threads++) {
        SmithWaterman sw(seq1, seq2);

        double start = omp_get_wtime();

        if (num_threads == 1) {
            sw.computeSerial();
        } else {
            sw.computeParallel(num_threads);
        }

        double end = omp_get_wtime();
        double elapsed = end - start;

        int max_score = sw.getMaxScore();

        if (num_threads == 1) {
            serial_time = elapsed;
            baseline_score = max_score;
        }

        double speedup = serial_time / elapsed;

        cout << fixed << setprecision(6);
        cout << "Threads: " << setw(2) << num_threads
             << " | Time: " << setw(10) << elapsed
             << " s | Speedup: " << setw(8) << speedup
             << " | Score: " << setw(6) << max_score << endl;

        results << num_threads << "\t" << elapsed << "\t" << speedup << "\t" << max_score << "\n";
    }

    results.close();
    cout << "\nResults saved to smithwaterman_results.txt\n";

    return 0;
}
