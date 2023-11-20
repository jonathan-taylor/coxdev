
#include <iostream>

// Compute cumsum with a padding of 0 at the beginning
// @param sequence input sequence [ro]
// @param output output sequence  [w]
template <class T>
void forward_cumsum(const std::vector<T> sequence,
		       std::vector<T> output)
{
  T sum = (T) 0;
  output[0] = sum;
  auto j = std::next(output.begin());
  for (auto i = sequence.begin(); i != sequence.end(); ++i, ++j) {
    sum += sequence[i];
    output[j] = sum;
  }
}


int main(int argc, char **argv) {
  std::vector<int> myVector = {1, 2, 3, 4, 5};
  std::vector<int> outVector(5);
  forward_cumsum<int>(myVector, outVector);
  for (auto i = outVector.begin(); i != outVector.end(); i++) {
    printf("%d ", outVector(i));
  }
  printf("\n");
  return(0);
}
