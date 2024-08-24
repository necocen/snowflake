use ndarray::{Array2, Zip};
use std::collections::HashMap;

type Point = (f32, f32);

fn init_grid(n: usize, beta: f32) -> Array2<f32> {
    let mut s = Array2::<f32>::ones((n, n)) * beta;
    s[[n / 2, n / 2]] = 1.0;
    s
}

fn update_grid(n: usize, s: Array2<f32>, gamma: f32, alpha: f32) -> Array2<f32> {
    let mut receptive_cells = Array2::<bool>::default((n, n));
    Zip::indexed(&mut receptive_cells).par_for_each(|(i, j), cell| {
        *cell = s[[i, j]] >= 1.0
            || s[[(i + 1) % n, j]] >= 1.0
            || s[[(i + n - 1) % n, j]] >= 1.0
            || s[[i, (j + 1) % n]] >= 1.0
            || s[[i, (j + n - 1) % n]] >= 1.0
            || s[[(i + n - 1) % n, (j + 1) % n]] >= 1.0
            || s[[(i + 1) % n, (j + n - 1) % n]] >= 1.0;
    });

    let u0 = Zip::from(&s).and(&receptive_cells).map_collect(
        |&s_val, &receptive| {
            if receptive {
                0.0
            } else {
                s_val
            }
        },
    );
    let v0 = Zip::from(&s).and(&receptive_cells).map_collect(
        |&s_val, &receptive| {
            if receptive {
                s_val
            } else {
                0.0
            }
        },
    );
    let mut u1 = Array2::<f32>::zeros((n, n));
    let mut v1 = v0.clone();

    // Rule 1
    Zip::indexed(&mut v1).par_for_each(|(i, j), cell| {
        if receptive_cells[[i, j]] {
            *cell += gamma;
        }
    });

    // Rule 2
    Zip::indexed(&mut u1).par_for_each(|(i, j), cell| {
        let mut u0_neighbors = 0.0;
        u0_neighbors += u0[[(i + 1) % n, j]];
        u0_neighbors += u0[[(i + n - 1) % n, j]];
        u0_neighbors += u0[[i, (j + 1) % n]];
        u0_neighbors += u0[[i, (j + n - 1) % n]];
        u0_neighbors += u0[[(i + n - 1) % n, (j + 1) % n]];
        u0_neighbors += u0[[(i + 1) % n, (j + n - 1) % n]];
        u0_neighbors /= 6.0;
        *cell = u0[[i, j]] + alpha * (u0_neighbors - u0[[i, j]]) / 2.0;
    });

    // Update s
    &u1 + &v1
}

fn extract_contours(grid: &Array2<bool>, a: f32) -> Vec<Vec<Point>> {
    let n = grid.shape()[0];
    let mut segments: HashMap<(i32, i32), (i32, i32)> = HashMap::new();

    let directions = [(1, 1), (0, 2), (-1, 1), (-1, -1), (0, -2), (1, -1)];

    for i in 0..n {
        for j in 0..n {
            if grid[[i, j]] {
                for (k, &(di, dj)) in directions.iter().enumerate() {
                    let start = (2 * i as i32 + j as i32 + di, 3 * j as i32 + dj);
                    let end = (
                        2 * i as i32 + j as i32 + directions[(k + 1) % 6].0,
                        3 * j as i32 + directions[(k + 1) % 6].1,
                    );

                    if let Some(&reverse_start) = segments.get(&end) {
                        if reverse_start == start {
                            segments.remove(&end);
                        } else {
                            segments.insert(start, end);
                        }
                    } else {
                        segments.insert(start, end);
                    }
                }
            }
        }
    }

    let mut contours = Vec::new();
    while !segments.is_empty() {
        let start = *segments.keys().next().unwrap();
        let mut contour = vec![start];
        let mut current = start;
        while let Some(&next) = segments.get(&current) {
            contour.push(next);
            segments.remove(&current);
            current = next;
        }
        contours.push(contour);
    }

    contours
        .into_iter()
        .map(|contour| {
            contour
                .into_iter()
                .map(|(x, y)| (a * x as f32 / 2.0, a * y as f32 / 2.0 / 3.0f32.sqrt()))
                .collect()
        })
        .collect()
}

fn main() {
    let n = 500;
    let alpha = 0.502;
    let beta = 0.4;
    let gamma = 0.0001;
    let num_steps = 10000;

    let mut s = init_grid(n, beta);
    for _ in 0..num_steps {
        s = update_grid(n, s, gamma, alpha);
    }

    let grid = s.mapv(|x| x >= 1.0);
    let contours = extract_contours(&grid, 1.0);

    println!("Number of contours: {}", contours.len());
}
