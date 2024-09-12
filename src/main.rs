use std::{fs::OpenOptions, sync::Arc};

use bevy::{prelude::*, window::PrimaryWindow};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use ndarray::Array2;
use parking_lot::RwLock;
use svg::node::element::{path::Data, Path, SVG};

mod gravner_griffeath;
mod reiter;
mod stl;
mod utils;
mod visualization;

fn main() {
    App::new()
        .init_resource::<Field>()
        .add_event::<ResetSimulation>()
        .add_plugins((DefaultPlugins, EguiPlugin))
        // .add_plugins(reiter::ReiterSimulatorPlugin)
        .add_plugins(gravner_griffeath::GravnerGrifeeathSimulatorPlugin)
        .add_plugins(visualization::VisualizationPlugin)
        .add_systems(Startup, (start_simulation, set_window_title))
        .add_systems(Update, configure_ui)
        .run();
}

fn start_simulation(field: Res<Field>) {
    let mut field = field.0.write();
    field.is_running = true;
}

#[derive(Event, Default)]
struct ResetSimulation;

#[derive(Resource, Default)]
pub struct Field(pub Arc<RwLock<FieldInner>>);

pub struct FieldInner {
    pub cells: Array2<f32>,
    pub step: u64,
    pub is_running: bool,
}

impl FieldInner {
    fn new(n: usize) -> Self {
        Self {
            cells: Array2::<f32>::zeros((n, n)),
            step: 0,
            is_running: false,
        }
    }

    pub fn write_to_svg(&self) -> SVG {
        let n = self.cells.shape()[0];
        let width = 1.5 * n as f32;
        let height = f32::sqrt(3.0) * n as f32 / 2.0;
        let contours = utils::extract_contours(&self.cells.mapv(|c| c > 0.0), 1.0);
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
        document
    }

    pub fn write_to_stl(&self) -> std::io::Result<()> {
        let facets = stl::cells_to_facets(&self.cells, 0.3, 1.5);
        let triangles = stl::calculate_normal(facets);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open("mesh.stl")?;
        stl_io::write_stl(&mut file, triangles.iter())
    }
}

impl Default for FieldInner {
    fn default() -> Self {
        Self::new(1000)
    }
}

fn configure_ui(
    mut contexts: EguiContexts,
    field: Res<Field>,
    mut reset_events: EventWriter<ResetSimulation>,
) {
    egui::Window::new("Control").show(contexts.ctx_mut(), |ui| {
        let FieldInner {
            is_running, step, ..
        } = *field.0.read();
        ui.add(egui::Label::new(format!("Step: {}", step)));
        ui.horizontal(|ui| {
            {
                if ui
                    .button(if is_running { "Pause" } else { "Resume" })
                    .clicked()
                {
                    let mut field = field.0.write();
                    field.is_running = !field.is_running;
                }
                if ui.button("Save").clicked() {
                    // if let Err(e) = svg::save("snowflake.svg", &field.0.read().write_to_svg()) {
                    //     tracing::error!("Failed to save SVG: {}", e);
                    // } else {
                    //     tracing::info!("Saved SVG");
                    // }
                    if let Err(e) = field.0.read().write_to_stl() {
                        tracing::error!("Failed to save STL: {}", e);
                    } else {
                        tracing::info!("Saved STL");
                    }
                }
            }
            if ui.button("Reset").clicked() {
                reset_events.send(ResetSimulation);
                tracing::info!("Reset");
            }
        });
    });
}

fn set_window_title(mut window_query: Query<&mut Window, With<PrimaryWindow>>) {
    if let Ok(mut window) = window_query.get_single_mut() {
        window.title = "Snowflake Simulator".to_string();
    }
}
