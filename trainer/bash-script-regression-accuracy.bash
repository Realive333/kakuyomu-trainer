echo "regression accuracy shell script start"

target=( 1 2 3 4 5 20 39 40 42 69 70 71 73 74 75 77 79 80 81 83 84 87 90 96 120 121 122 126 128 199 200 203 204 214 259 260 281 284 291)
model="LinearRegression"

for item in "${target[@]}";
do
  echo "Counting... target=$item"
  python get-regression-accuracy.py --target=$item --model=$model
done