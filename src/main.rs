use bevy::{
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use fnv::FnvHashMap;
use ndarray::Array2;
use simulation::{init_grid, update_grid};

mod simulation;

#[derive(Resource)]
struct Field {
    cells: Array2<f32>,
    meshes: FnvHashMap<(usize, usize), Entity>,
    step: u64,
    scale: f32,
}

impl Default for Field {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl Field {
    pub fn new(n: usize) -> Self {
        Self {
            cells: Array2::<f32>::zeros((n, n)),
            meshes: FnvHashMap::default(),
            step: 0,
            scale: 4.0,
        }
    }

    pub fn init(&mut self, beta: f32) {
        self.cells = init_grid(self.n(), beta);
        self.step = 0;
    }

    pub fn update(&mut self, alpha: f32, gamma: f32) {
        update_grid(self.n(), &mut self.cells, gamma, alpha);
        self.step += 1;
    }

    pub fn n(&self) -> usize {
        self.cells.shape()[0]
    }

    pub fn transition_at(&self, x: usize, y: usize) -> Vec3 {
        let n = self.n();
        Vec3::new(
            x as f32 + y as f32 / 2.0 - n as f32 * 0.75,
            (y as f32 - (n / 2) as f32) * f32::sqrt(3.0) / 2.0,
            0.0,
        ) * self.scale
    }
}

#[derive(Resource)]
struct SimulationConfig {
    alpha: f32,
    beta: f32,
    gamma: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            alpha: 0.502,
            beta: 0.4,
            gamma: 0.0001,
        }
    }
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins))
        .init_resource::<Field>()
        .init_resource::<SimulationConfig>()
        .add_systems(Startup, setup)
        .add_systems(PostStartup, init_simulation)
        .add_systems(Update, update_visualization)
        .add_systems(FixedUpdate, update_simulation)
        .insert_resource(Time::<Fixed>::from_hz(200f64))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut field: ResMut<Field>,
) {
    commands.spawn(Camera2dBundle::default());
    let n = field.n();

    let hexagon = Mesh2dHandle(meshes.add(RegularPolygon::new(field.scale / f32::sqrt(3.0), 6)));
    let color = materials.add(Color::WHITE);

    for i in 0..n {
        for j in 0..n {
            if let Some(id) = field.meshes.get(&(i, j)) {
                commands.entity(*id).despawn();
            }
            let id = commands
                .spawn(MaterialMesh2dBundle {
                    visibility: Visibility::Hidden,
                    mesh: hexagon.clone(),
                    material: color.clone(),
                    transform: Transform::from_translation(field.transition_at(i, j)),
                    ..default()
                })
                .id();
            field.meshes.insert((i, j), id);
        }
    }
}

fn init_simulation(mut field: ResMut<Field>, config: Res<SimulationConfig>) {
    field.init(config.beta);
}

fn update_simulation(mut field: ResMut<Field>, config: Res<SimulationConfig>) {
    field.update(config.alpha, config.gamma);
}

fn update_visualization(mut commands: Commands, field: Res<Field>) {
    let frozen_cells = field.cells.map(|&x| x >= 1.0);
    // let mut contours = extract_contours(&threshold_grid, sim_state.scale);
    // FIXME: 輪郭が長いものが必ず外側にくるとは（厳密には）限らないはずだが、だいたいはそうなので一旦これで
    // contours.sort_by(|a, b| b.len().cmp(&a.len()));
    let n = field.n();
    for i in 0..n {
        for j in 0..n {
            let id = field.meshes.get(&(i, j)).unwrap();
            if frozen_cells[[i, j]] {
                commands.entity(*id).insert(Visibility::Visible);
            } else {
                commands.entity(*id).insert(Visibility::Hidden);
            }
        }
    }
}
