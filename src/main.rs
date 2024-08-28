use std::path::PathBuf;

use bevy::{
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use fnv::FnvHashMap;
use ndarray::Array2;
use simulation::{extract_contours, init_grid, update_grid};
use svg::node::element::{path::Data, Path};

mod simulation;

#[derive(Resource)]
struct Field {
    cells: Array2<f32>,
    meshes: FnvHashMap<(usize, usize), Entity>,
    step: u64,
    scale: f32,
    is_running: bool,
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
            scale: 1.0,
            is_running: false,
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

    pub fn write_to_svg(&self, filepath: impl Into<PathBuf>) -> Result<(), std::io::Error> {
        let n = self.n();
        let width = 1.5 * n as f32;
        let height = f32::sqrt(3.0) * n as f32 / 2.0;
        let frozen_cells = self.cells.map(|&x| x >= 1.0);
        let contours = extract_contours(&frozen_cells, 1.0);
        let mut data = Data::new();
        for contour in contours {
            let mut iter = contour.iter();
            if let Some((x, y)) = iter.next() {
                data = data.move_to((*x, *y));
            }
            for (x, y) in iter {
                data = data.line_to((*x, *y));
            }
            data = data.close();
        }
        let path = Path::new().set("fill", "black").set("d", data);

        let document = svg::Document::new()
            .set("viewBox", (0, 0, width, height))
            .set("width", width)
            .set("height", height)
            .add(path);
        svg::save(filepath.into(), &document)
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
        .add_plugins((DefaultPlugins, EguiPlugin))
        .init_resource::<Field>()
        .init_resource::<SimulationConfig>()
        .add_systems(Startup, setup)
        .add_systems(PostStartup, init_simulation)
        .add_systems(Update, (configure_ui, update_visualization))
        .add_systems(FixedUpdate, update_simulation)
        .insert_resource(Time::<Fixed>::from_hz(100f64))
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
    field.is_running = true;
}

fn update_simulation(mut field: ResMut<Field>, config: Res<SimulationConfig>) {
    if field.is_running {
        field.update(config.alpha, config.gamma);
    }
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

fn configure_ui(
    mut contexts: EguiContexts,
    mut field: ResMut<Field>,
    mut config: ResMut<SimulationConfig>,
) {
    egui::Window::new("Snowflake").show(contexts.ctx_mut(), |ui| {
        ui.add(egui::Label::new(format!("Step: {}", field.step)));
        ui.vertical(|ui| {
            ui.add(egui::Slider::new(&mut config.alpha, 0.0..=2.0).text("alpha"));
            // TODO: 実行中に背景場を変えられるような実装が必要で、それはメソッドにしないといけない
            ui.add(egui::Slider::new(&mut config.beta, 0.0..=1.0).text("beta"));
            ui.add(
                egui::Slider::new(&mut config.gamma, 0.0..=1.0)
                    .text("gamma")
                    .logarithmic(true),
            );
        });
        ui.horizontal(|ui| {
            if ui
                .button(if field.is_running { "Pause" } else { "Resume" })
                .clicked()
            {
                field.is_running = !field.is_running;
            }
            if ui.button("Save").clicked() {
                field.write_to_svg("snowflake.svg").unwrap();
            }
            if ui.button("Reset").clicked() {
                field.init(config.beta);
                field.is_running = true;
            }
        });
    });
}
