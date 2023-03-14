echo "baseline-full shell script start"

dataset='first-match-character-scatter'
target=( 1 2 3 4 5 20 39 40 42 69 70 71 73 74 75 77 79 80 81 83 84 87 90 96 120 121 122 126 128 199 200 203 204 214 259 260 281 284 291)

for item in "${target[@]}";
do
  echo "SVM training... dataset=$dataset target=$item"
  python baseline-svm-complete.py --dataset $dataset --target $item
done

for item in "${target[@]}";
do
  echo "BNB training... dataset=$dataset target=$item"
  python baseline-bnb-complete.py --dataset $dataset --target $item
done

for item in "${target[@]}";
do
  echo "LR training... dataset=$dataset target=$item"
  python baseline-lr-complete.py --dataset $dataset --target $item
done

for item in "${target[@]}";
do
  echo "MNB training... dataset=$dataset target=$item"
  python baseline-mnb-complete.py --dataset $dataset --target $item
done