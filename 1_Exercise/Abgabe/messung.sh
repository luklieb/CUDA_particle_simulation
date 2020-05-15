#!/bin/bash


OUTFILE=data.txt
THREADS=`seq 2 2 32`
PROGS='CPU CUDA CL'

make

#rm -f $OUTFILE
echo "starting"

for i in $PROGS
do
	echo "$i-float" >> $OUTFILE
	for j in $THREADS
	do
		name="julia$i"
		tmp=$(./$name $j $j)
		tmp2=$(echo $tmp | grep invalid)
		echo "tmp= $tmp"
		echo "tmp2= $tmp2"
		if [ "$tmp2" = "" ] 
		then
			echo "if: $j $tmp"
			echo "$j $tmp" >> $OUTFILE
		fi
	done
	echo "" >> $OUTFILE
	echo "" >> $OUTFILE
	echo "" >> $OUTFILE
	echo "" >> $OUTFILE
	echo "" >> $OUTFILE
done

echo "" >> $OUTFILE
echo "" >> $OUTFILE
echo "" >> $OUTFILE
echo "" >> $OUTFILE
echo "" >> $OUTFILE
