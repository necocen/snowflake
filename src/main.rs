use bevy::prelude::*;
use bevy_prototype_lyon::prelude::*;
use ndarray::Array2;
use simulation::{extract_contours, init_grid, update_grid};

mod simulation;

#[derive(Resource)]
struct SimulationState {
    grid: Array2<f32>,
    n: usize,
    alpha: f32,
    #[allow(dead_code)]
    beta: f32,
    gamma: f32,
    scale: f32,
    step: u64,
}

#[derive(Component)]
struct Snowflake;

impl SimulationState {
    pub fn new(n: usize, alpha: f32, beta: f32, gamma: f32, scale: f32) -> Self {
        Self {
            grid: init_grid(n, beta),
            n,
            alpha,
            beta,
            gamma,
            scale,
            step: 0,
        }
    }
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, ShapePlugin))
        .insert_resource(SimulationState::new(1000, 0.502, 0.4, 0.0001, 2.0))
        .add_systems(Startup, setup)
        .add_systems(Update, update_visualization)
        .add_systems(FixedUpdate, update_simulation)
        .insert_resource(Time::<Fixed>::from_hz(250f64))
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
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
    let mut contours = extract_contours(&threshold_grid, sim_state.scale);
    // FIXME: 輪郭が長いものが必ず外側にくるとは（厳密には）限らないはずだが、だいたいはそうなので一旦これで
    contours.sort_by(|a, b| b.len().cmp(&a.len()));
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
