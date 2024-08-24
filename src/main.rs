use bevy::prelude::*;
use bevy_prototype_lyon::prelude::*;
use ndarray::{Array2, Zip};
use std::collections::HashMap;

type Point = (f32, f32);

fn init_grid(n: usize, beta: f32) -> Array2<f32> {
    let mut s = Array2::<f32>::ones((n, n)) * beta;
    s[[n / 2, n / 2]] = 1.0;
    s
}

fn update_grid(n: usize, s: &mut Array2<f32>, gamma: f32, alpha: f32) {
    let s_view = s.view();
    // receptive_cells と uv0 を同時に計算
    let combined = Zip::indexed(s.view()).par_map_collect(|(i, j), &s_val| {
        let receptive = s_val >= 1.0
            || s_view[[(i + 1) % n, j]] >= 1.0
            || s_view[[(i + n - 1) % n, j]] >= 1.0
            || s_view[[i, (j + 1) % n]] >= 1.0
            || s_view[[i, (j + n - 1) % n]] >= 1.0
            || s_view[[(i + n - 1) % n, (j + 1) % n]] >= 1.0
            || s_view[[(i + 1) % n, (j + n - 1) % n]] >= 1.0;
        let uv = if receptive {
            (0.0, s_val)
        } else {
            (s_val, 0.0)
        };
        (receptive, uv)
    });

    // s の更新を行う
    Zip::indexed(s)
        .and(&combined)
        .par_for_each(|(i, j), s_val, &(receptive, (u0, v0))| {
            let mut v1 = v0;
            // Rule 1
            if receptive {
                v1 += gamma;
            }

            // Rule 2
            let mut u0_neighbors = 0.0;
            for (di, dj) in &[(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, -1)] {
                let (ni, nj) = (
                    (i as isize + di + n as isize) as usize % n,
                    (j as isize + dj + n as isize) as usize % n,
                );
                u0_neighbors += combined[[ni, nj]].1 .0;
            }
            u0_neighbors /= 6.0;

            let u1 = u0 + alpha * (u0_neighbors - u0) / 2.0;
            *s_val = u1 + v1;
        });
}

fn extract_contours(grid: &Array2<bool>, scale: f32) -> Vec<Vec<Point>> {
    let mut segments: HashMap<(i32, i32), (i32, i32)> = HashMap::new();
    let directions = [(1, 1), (0, 2), (-1, 1), (-1, -1), (0, -2), (1, -1)];
    grid.indexed_iter().for_each(|((i, j), v)| {
        if *v {
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
    });

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
                .map(|(x, y)| {
                    (
                        scale * x as f32 / 2.0,
                        scale * y as f32 / 2.0 / 3.0f32.sqrt(),
                    )
                })
                .collect()
        })
        .collect()
}

#[derive(Resource)]
struct SimulationState {
    grid: Array2<f32>,
    n: usize,
    beta: f32,
    gamma: f32,
    alpha: f32,
    scale: f32,
    step: u64,
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, ShapePlugin))
        .insert_resource(SimulationState {
            grid: Array2::zeros((0, 0)),
            n: 1000,
            beta: 0.4,
            gamma: 0.0001,
            alpha: 0.502,
            scale: 2.0,
            step: 0,
        })
        .add_systems(Startup, setup)
        .add_systems(Update, update_visualization)
        .add_systems(FixedUpdate, update_simulation)
        .insert_resource(Time::<Fixed>::from_hz(1000f64))
        .run();
}

fn setup(mut commands: Commands, mut sim_state: ResMut<SimulationState>) {
    commands.spawn(Camera2dBundle::default());
    sim_state.grid = init_grid(sim_state.n, sim_state.beta);
}

fn update_simulation(mut sim_state: ResMut<SimulationState>) {
    let gamma = sim_state.gamma;
    let alpha = sim_state.alpha;
    update_grid(sim_state.n, &mut sim_state.grid, gamma, alpha);
    sim_state.step += 1;
}

fn update_visualization(
    mut commands: Commands,
    sim_state: Res<SimulationState>,
    query: Query<Entity, With<Snowflake>>,
) {
    // 既存の雪の結晶を削除
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }

    // 新しい輪郭を描画
    let threshold_grid = sim_state.grid.map(|&x| x >= 1.0);
    let contours = extract_contours(&threshold_grid, sim_state.scale);
    tracing::info!("step {}: {}", sim_state.step, contours.len());

    for (i, contour) in contours.iter().enumerate() {
        let mesh = shapes::Polygon {
            points: contour
                .iter()
                .map(|&(x, y)| {
                    Vec2::new(
                        x - sim_state.scale * sim_state.n as f32 / 2.0 - 800.0,
                        y - sim_state.scale * sim_state.n as f32 / 2.0,
                    )
                })
                .collect(),
            closed: true,
        };
        let color = if i == 0 { Color::WHITE } else { Color::BLACK };

        commands.spawn((
            ShapeBundle {
                path: GeometryBuilder::build_as(&mesh),
                ..default()
            },
            Fill::color(color),
        ));
    }
}

#[derive(Component)]
struct Snowflake;
