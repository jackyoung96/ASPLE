# FILES=$(find trck_features/OA -name "*.csv")
# for i in $FILES ; do
#   echo $i
#   python tracking_test.py --input=$i --epsilon=0.3
# done

FILES=$(find cls_features/SA2/left -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test.py --input=$i  
done

FILES=$(find cls_features/SA2/right -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test.py --input=$i  
done

FILES=$(find cls_features/SB1/left -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test.py --input=$i  
done

FILES=$(find cls_features/SB1/right -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test.py --input=$i  
done

FILES=$(find cls_features/SB2/left -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test.py --input=$i  
done

FILES=$(find cls_features/SB2/right -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test.py --input=$i  
done
