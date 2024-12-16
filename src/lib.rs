use bevy::{prelude::*, window::PrimaryWindow};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use chrono::{DateTime, Local};
use ndarray::Array2;
#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

mod gravner_griffeath;
#[cfg(not(target_family = "wasm"))]
mod png;
mod reiter;
#[cfg(not(target_family = "wasm"))]
mod stl;
#[cfg(not(target_family = "wasm"))]
mod svg;
mod visualization;

#[cfg(target_family = "wasm")]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub fn run() {
    App::new()
        .init_resource::<Field>()
        .add_event::<ControlEvent>()
        .add_plugins((DefaultPlugins, EguiPlugin))
        // .add_plugins(reiter::ReiterSimulatorPlugin)
        .add_plugins(gravner_griffeath::GravnerGriffeathSimulatorPlugin)
        .add_plugins(visualization::VisualizationPlugin)
        .add_systems(Startup, (start_simulation, set_window_title))
        .add_systems(Update, configure_ui)
        .run();
}

fn start_simulation(mut field: ResMut<Field>) {
    field.is_running = true;
}

#[allow(dead_code)]
#[derive(Event)]
enum ControlEvent {
    Reset,
    Save(DateTime<Local>),
}

#[derive(Resource)]
pub struct Field {
    pub cells: Array2<f32>,
    pub step: u64,
    pub is_running: bool,
}

impl Field {
    fn new(n: usize) -> Self {
        Self {
            cells: Array2::<f32>::zeros((n, n)),
            step: 0,
            is_running: false,
        }
    }
}

impl Default for Field {
    fn default() -> Self {
        Self::new(1000)
    }
}

fn configure_ui(
    mut contexts: EguiContexts,
    mut field: ResMut<Field>,
    mut events: EventWriter<ControlEvent>,
) {
    egui::Window::new("Control").show(contexts.ctx_mut(), |ui| {
        ui.add(egui::Label::new(format!("Step: {}", field.step)));
        ui.horizontal(|ui| {
            {
                if ui
                    .button(if field.is_running { "Pause" } else { "Resume" })
                    .clicked()
                {
                    field.is_running = !field.is_running;
                }
                cfg_if::cfg_if! {
                    if #[cfg(not(target_family = "wasm"))] {
                        if ui.button("Save STL").clicked() {
                            let now = Local::now();
                            events.send(ControlEvent::Save(now));
                            match stl::write_to_stl(&field.cells, now) {
                                Ok(path) => {
                                    tracing::info!("Saved STL: {}", path.display());
                                }
                                Err(e) => {
                                    tracing::error!("Failed to save STL: {e}");
                                }
                            }
                        }
                        if ui.button("Save SVG").clicked() {
                            let now = Local::now();
                            events.send(ControlEvent::Save(now));
                            match svg::write_to_svg(&field.cells, now) {
                                Ok(path) => {
                                    tracing::info!("Saved SVG: {}", path.display());
                                }
                                Err(e) => {
                                    tracing::error!("Failed to save SVG: {e}");
                                }
                            }
                        }
                        if ui.button("Save PNG").clicked() {
                            let now = Local::now();
                            events.send(ControlEvent::Save(now));
                            match png::write_to_png(&field.cells, now) {
                                Ok(path) => {
                                    tracing::info!("Saved PNG: {}", path.display());
                                }
                                Err(e) => {
                                    tracing::error!("Failed to save PNG: {e}");
                                }
                            }
                        }
                    }
                }
            }
            if ui.button("Reset").clicked() {
                events.send(ControlEvent::Reset);
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
