FILES=$(find cls_features -name "*.wav")
for i in $FILES ; do
  echo $i
  python featureExtract.py --input=$i
done