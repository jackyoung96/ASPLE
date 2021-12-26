
FILES=$(find trck_features/OA -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test.py --input=$i --epsilon=0.3
done