use ndarray::*;
use ndarray_linalg::*;
use polars::prelude::*;
use plotters::prelude::*;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 读取 CSV 文件
    let file = File::open("ps.csv")?;
    let df = CsvReader::new(file)
        .has_header(true)
        .finish()?;

    // 检查并处理列名（根据实际 CSV 调整）
    let columns = df.get_column_names();
    println!("检测到的列: {:?}", columns);

    // 提取数据列（假设列顺序为 t, x, y, z）
    let t = df.column("t")?.f64()?;
    let x = df.column("x")?.f64()?;
    let y = df.column("y")?.f64()?; // 如果列名不同需要修改
    let z = df.column("z")?.f64()?;

    // 转换为 Vec<f64>
    let t_data: Vec<f64> = t.into_no_null_iter().collect();
    let x_data = Array1::from_vec(x.into_no_null_iter().collect());
    let y_data = Array1::from_vec(y.into_no_null_iter().collect());
    let z_data = Array1::from_vec(z.into_no_null_iter().collect());

    // 为每个坐标轴执行拟合
    let (coeff_x, cov_x) = polynomial_fit(&t_data, &x_data, 4)?;
    let (coeff_y, cov_y) = polynomial_fit(&t_data, &y_data, 4)?;
    let (coeff_z, cov_z) = polynomial_fit(&t_data, &z_data, 4)?;

    // 打印结果
    print_equation("x", &coeff_x);
    print_equation("y", &coeff_y);
    print_equation("z", &coeff_z);

    // 绘制协方差矩阵
    plot_matrix(&cov_x, "covariance_x.png")?;
    plot_matrix(&cov_y, "covariance_y.png")?;
    plot_matrix(&cov_z, "covariance_z.png")?;

    Ok(())
}

fn polynomial_fit(
    t: &[f64],
    y: &Array1<f64>,
    degree: usize,
) -> Result<(Array1<f64>, Array2<f64>), Box<dyn std::error::Error>> {
    // 构建设计矩阵
    let mut x = Array2::zeros((t.len(), degree + 1));
    for (i, &ti) in t.iter().enumerate() {
        for j in 0..=degree {
            x[(i, j)] = ti.powi((degree - j) as i32);
        }
    }

    // 计算正规方程
    let xt = x.t();
    let xtx = xt.dot(&x);
    let xty = xt.dot(y);

    // 求逆并计算系数
    let xtx_inv = xtx.inv()?;
    let coefficients = xtx_inv.dot(&xty);

    // 计算协方差矩阵
    let residuals = y - x.dot(&coefficients);
    let sigma_sq = residuals.dot(&residuals) / (t.len() - degree - 1) as f64;
    let covariance = &xtx_inv * sigma_sq;

    Ok((coefficients, covariance))
}

fn print_equation(name: &str, coeff: &Array1<f64>) {
    let terms = coeff
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let power = 4 - i;
            match power {
                0 => format!("{:.4}", c),
                1 => format!("{:.4}t", c),
                _ => format!("{:.4}t^{}", c, power),
            }
        })
        .collect::<Vec<_>>()
        .join(" + ");

    println!("{} = {}", name, terms);
}

fn plot_matrix(matrix: &Array2<f64>, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (600, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("协方差矩阵", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..matrix.ncols(), 0..matrix.nrows())?;

    chart.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;

    let max_val = matrix.fold(f64::NEG_INFINITY, |a, &b| a.max(b.abs()));
    let min_val = -max_val;

    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let value = matrix[(i, j)];
            let color = get_color_normalized(value, min_val, max_val);

            chart.draw_series(std::iter::once(
                Rectangle::new([(j, i), (j + 1, i + 1)], color.filled())
            ))?;
        }
    }

    Ok(())
}

fn get_color_normalized(value: f64, min_val: f64, max_val: f64) -> RGBColor {
    if max_val == min_val {
        return RGBColor(128, 128, 128); // Gray if all values are the same
    }

    let normalized_value = (value - min_val) / (max_val - min_val);
    let r = (255.0 * (1.0 - normalized_value)) as u8;
    let g = (255.0 * normalized_value) as u8;
    let b = 0;

    RGBColor(r, g, b)
}



