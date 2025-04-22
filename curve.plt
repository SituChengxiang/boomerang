# 不设置终端，使用默认交互式终端
set datafile separator ","
unset dgrid3d  # 明确关闭插值
set hidden3d
set ticslevel 0
set view equal xyz

# 设置坐标轴不同颜色
set xtics textcolor rgb "red"
set ytics textcolor rgb "green"
set ztics textcolor rgb "blue"
set xlabel "X" textcolor rgb "red"
set ylabel "Y" textcolor rgb "green"
set zlabel "Z" textcolor rgb "blue"

# 手动设置坐标轴范围（根据实际数据调整）
set xrange [0.07:1.67]
set yrange [0.75:2.99]
set zrange [0.04:2.15]

# 打印CSV文件的前几行以进行检查
print "First few lines of ps1.csv:"
!head -n 5 ps1.csv

# 绘图，此时会在交互式窗口中显示，可以旋转拖动
splot 'ps1.csv' skip 1 using 1:2:3 with points pt 7 ps 1.0 lc rgb "purple" title 'Data Points'

pause mouse close "Click on the plot to close it."
