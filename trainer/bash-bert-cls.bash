echo "bash bert cls classification"

target=( 1 2 3 4 5 20 39 40 42 69 70 71 73 74 75 77 79 80 81 83 84 87 90 96 120 121 122 126 128 199 200 203 204 214 259 260 281 284 291)

for item in "${target[@]}";
do
  echo "predicting:... target=$item batchsize=5"
  python bert-cls.py --target=$item --batchsize=5
done
