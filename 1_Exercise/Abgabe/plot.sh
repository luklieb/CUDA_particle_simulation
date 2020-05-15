set title "Threads vs. Sec"
set xlabel "Threads"
set ylabel "Sec"
set autoscale
set logscale y
set key outside
set style line 1 lt rgb "blue"
set style line 2 lt rgb "red"
set style line 3 lt rgb "green"
set style line 4 lt rgb "0x00CED1"
set style line 5 lt rgb "0xFF6347"
set style line 6 lt rgb "0xADFF2F"


plot for [IDX=0:5] 'data.txt' i IDX u 1:2 w lines ls IDX+1 title columnhead


pause -1
