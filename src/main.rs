use polars::prelude::*;
use ndarray::{Array2, Array1};
use ndarray_linalg::{SVD, Solve};
use std::f64;

fn read_csv(file_path: &str) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    let df = CsvReader::from_path(file_path)?
        .infer_schema(None)
        .has_header(true)
        .finish()?;

    let t_series = df.column("t")?.f64()?;
    let x_series = df.column("x")?.f64()?;
    let y_series = df.column("y")?.f64()?;
    let z_series = df.column("z")?.f64()?;

    let t: Vec<f64> = t_series.into_iter().collect::<Option<Vec<_>>>().unwrap();
    let x: Vec<f64> = x_series.into_iter().collect::<Option<Vec<_>>>().unwrap();
    let y: Vec<f64> = y_series.into_iter().collect::<Option<Vec<_>>>().unwrap();
    let z: Vec<f64> = z_series.into_iter().collect::<Option<Vec<_>>>().unwrap();

    Ok((t, x, y, z))
}


fn solve_parameters(t: &[f64], data: &[f64], model_type: &str) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    let n = t.len();
    let (mut a, b) = match model_type {
        "x" => {
            let mut a_x = Array2::<f64>::zeros((n, 5));
            let mut b_x = Array1::<f64>::zeros(n);
            for i in 0..n {
                let ti = t[i];
                let xi = data[i];
                a_x[[i, 0]] = (1.0 - ti.cos()) * (-ti).exp(); // R (1-cos(ω)t)e^{-α t}
                a_x[[i, 1]] = ti * (-ti).exp();               // R cos(ω)t*e^{-α t}
                a_x[[i, 2]] = (-ti).exp();                    // Re^{-α t}
                a_x[[i, 3]] = ti;                             // v_{x0}t
                a_x[[i, 4]] = 1.0;                            // 常数项
                b_x[i] = xi;
            }
            (a_x, b_x)
        },
        "y" => {
            let mut a_y = Array2::<f64>::zeros((n, 4));
            let mut b_y = Array1::<f64>::zeros(n);
            for i in 0..n {
                let ti = t[i];
                let yi = data[i];
                a_y[[i, 0]] = ti.sin() * ti * (-ti).exp();     // R sin(ω)t*e^{-α t}
                a_y[[i, 1]] = ti * (-ti).exp();                // R cos(ω)t*e^{-α t}
                a_y[[i, 2]] = ti;                              // v_{y0}t
                a_y[[i, 3]] = 1.0;                             // 常数项
                b_y[i] = yi;
            }
            (a_y, b_y)
        },
        "z" => {
            let mut a_z = Array2::<f64>::zeros((n, 4));
            let mut b_z = Array1::<f64>::zeros(n);
            let g = 9.81; // 重力加速度
            for i in 0..n {
                let ti = t[i];
                let zi = data[i];
                a_z[[i, 0]] = ti;                              // v_{z0}t
                a_z[[i, 1]] = -0.5 * g * ti.powi(2);           // -\dfrac{1}{2}gt^2
                a_z[[i, 2]] = -ti.powi(2);                     // -β*v_{z0}t^2
                a_z[[i, 3]] = 1.0;                             // 常数项
                b_z[i] = zi;
            }
            (a_z, b_z)
        },
        _ => return Err("Invalid model type".into()),
    };

    // 使用 SVD::compute 正确调用
    let svd = a.clone().svd(true, true)?;
    let params = svd.solve(&b)?;

    Ok(params)
}

fn calculate_error(t: &[f64], data: &[f64], params: &Array1<f64>, model_type: &str) -> f64 {
    let n = t.len();
    let mut error = 0.0;
    for i in 0..n {
        let ti = t[i];
        let predicted = match model_type {
            "x" => {
                params[0] * (1.0 - ti.cos()) * (-ti).exp() +
                    params[1] * ti * (-ti).exp() +
                    params[2] * (-ti).exp() +
                    params[3] * ti +
                    params[4]
            },
            "y" => {
                params[0] * ti.sin() * ti * (-ti).exp() +
                    params[1] * ti * (-ti).exp() +
                    params[2] * ti +
                    params[3]
            },
            "z" => {
                params[0] * ti +
                    params[1] * -0.5 * 9.81 * ti.powi(2) +
                    params[2] * -ti.powi(2) +
                    params[3]
            },
            _ => continue,
        };
        error += (predicted - data[i]).powi(2);
    }
    (error / n as f64).sqrt()
}

fn print_results(model_type: &str, params: &Array1<f64>) {
    println!("Fitted parameters for {}(t):", model_type);
    match model_type {
        "x" => {
            println!("R(1-cos(ω)t)e^{{-α t}}: {}", params[0]);
            println!("Rcos(ω)t*e^{{-α t}}: {}", params[1]);
            println!("Re^{{-α t}}: {}", params[2]);
            println!("v_{{x0}}t: {}", params[3]);
            println!("Constant term: {}", params[4]);
        },
        "y" => {
            println!("Rsin(ω)t*e^{{-α t}}: {}", params[0]);
            println!("Rcos(ω)t*e^{{-α t}}: {}", params[1]);
            println!("v_{{y0}}t: {}", params[2]);
            println!("Constant term: {}", params[3]);
        },
        "z" => {
            println!("v_{{z0}}t: {}", params[0]);
            println!("-\\dfrac{{1}}{{2}}gt^2: {}", params[1]);
            println!("-β*v_{{z0}}t^2: {}", params[2]);
            println!("Constant term: {}", params[3]);
        },
        _ => (),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "ps.csv";
    let (t, x, y, z) = read_csv(file_path)?;

    let params_x = solve_parameters(&t, &x, "x")?;
    let params_y = solve_parameters(&t, &y, "y")?;
    let params_z = solve_parameters(&t, &z, "z")?;

    let error_x = calculate_error(&t, &x, &params_x, "x");
    let error_y = calculate_error(&t, &y, &params_y, "y");
    let error_z = calculate_error(&t, &z, &params_z, "z");

    print_results("x", &params_x);
    println!("Root Mean Square Error for x(t): {}", error_x);

    print_results("y", &params_y);
    println!("Root Mean Square Error for y(t): {}", error_y);

    print_results("z", &params_z);
    println!("Root Mean Square Error for z(t): {}", error_z);

    Ok(())
}



