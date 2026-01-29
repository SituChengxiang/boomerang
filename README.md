# 中国大学生物理学术竞赛（CUPT） Invent Yourself: Paper Boomerang 代码参考

CUPT让我们相遇于此吧（  
这个仓库里有：  
- Go语言的数值计算和GUI模拟
- Rust用于科学计算的尝试
- Python进行数据拟合尝试
- ~~Gunplot绘图代码~~
- Julia尝试更高性能计算
- ~~wxMaxima瞎写的代码~~

（我真是成分复杂啊……）

来看看就好，如果能帮到你，本人不胜荣幸

## 项目结构

因为把研究工作和开发工作放在了一个文件夹里，所以整个项目看起来并不那么常规
```plaintext
boomerang
|-- README.md
|-- data（数据组）
|   |-- interm（处理时的中间文件）
|   |-- raw（原始数据文件）
|   `-- final（处理后的文件）
|-- out（整个项目输出的一些文件）
`-- src（源码文件）
    |-- visualization（可视化相关）
    |-- fit（轨迹拟合相关）
    |-- preprocess（预处理）
    `-- utlis（一些小的工具）
```

## 我踩的坑

1. 一定要先确保处理后的数据是物理自洽的
2. 