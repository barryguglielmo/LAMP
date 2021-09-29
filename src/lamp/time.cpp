#include <iostream>
using namespace std;
#include <chrono>
using namespace std::chrono;

// Use auto keyword to avoid typing long
// type definitions to get the timepoint
// at this instant use function now()

int main() {
  auto start = std::chrono::system_clock::now();
  int i = 0;
  while (i < 1000000000)
    ++i;
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
}
