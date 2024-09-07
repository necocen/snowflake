use std::path::PathBuf;

use bevy::{
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use bevy_egui::{egui, EguiContexts};
use fnv::FnvHashMap;
use ndarray::{Array2, Zip};
use svg::node::element::{path::Data, Path};

use crate::utils;

pub struct ReiterSimulatorPlugin;

impl Plugin for ReiterSimulatorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ReiterField>();
        app.init_resource::<SimulationConfig>();
        app.add_systems(Startup, setup);
        app.add_systems(PostStartup, init_simulation);
        app.add_systems(Update, (configure_ui, update_visualization));
        app.add_systems(FixedUpdate, update_simulation);
        app.insert_resource(Time::<Fixed>::from_hz(200f64));
    }
}

#[derive(Resource)]
struct ReiterField {
    pub cells: Array2<f32>,
    pub meshes: FnvHashMap<(usize, usize), Entity>,
    pub step: u64,
    pub scale: f32,
    pub is_running: bool,
}

impl Default for ReiterField {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl ReiterField {
    fn new(n: usize) -> Self {
        Self {
            cells: Array2::<f32>::zeros((n, n)),
            meshes: FnvHashMap::default(),
            step: 0,
            scale: 1.0,
            is_running: false,
        }
    }

    fn init(&mut self, beta: f32) {
        self.cells = init_grid(self.n(), beta);
        self.step = 0;
    }

    fn update(&mut self, alpha: f32, gamma: f32) {
        update_grid(self.n(), &mut self.cells, gamma, alpha);
        self.step += 1;
    }

    fn n(&self) -> usize {
        self.cells.shape()[0]
    }

    fn transition_at(&self, x: usize, y: usize) -> Vec3 {
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
        let contours = utils::extract_contours(&frozen_cells, 1.0);
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
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
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

pub fn init_grid(n: usize, beta: f32) -> Array2<f32> {
    let mut s = Array2::<f32>::ones((n, n)) * beta;
    s[[n / 2, n / 2]] = 1.0;
    s
}

pub fn update_grid(n: usize, s: &mut Array2<f32>, gamma: f32, alpha: f32) {
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

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut field: ResMut<ReiterField>,
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

fn init_simulation(mut field: ResMut<ReiterField>, config: Res<SimulationConfig>) {
    field.init(config.beta);
    field.is_running = true;
}

fn update_simulation(mut field: ResMut<ReiterField>, config: Res<SimulationConfig>) {
    if field.is_running {
        field.update(config.alpha, config.gamma);
    }
}

fn update_visualization(mut commands: Commands, field: Res<ReiterField>) {
    let frozen_cells = field.cells.map(|&x| x >= 1.0);
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
    mut field: ResMut<ReiterField>,
    mut config: ResMut<SimulationConfig>,
) {
    egui::Window::new("Reiter's Snowflake").show(contexts.ctx_mut(), |ui| {
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
