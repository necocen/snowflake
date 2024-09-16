use std::sync::Arc;

use bevy::{prelude::*, window::PrimaryWindow};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use chrono::{DateTime, Local};
use ndarray::Array2;
use parking_lot::RwLock;

mod gravner_griffeath;
mod gravner_griffeath_wasm;
mod reiter;
mod stl;
mod visualization;

fn main() {
    App::new()
        .init_resource::<Field>()
        .add_event::<ControlEvent>()
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

#[derive(Event)]
enum ControlEvent {
    Reset,
    Save(DateTime<Local>),
}

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
}

impl Default for FieldInner {
    fn default() -> Self {
        Self::new(1000)
    }
}

fn configure_ui(
    mut contexts: EguiContexts,
    field: Res<Field>,
    mut events: EventWriter<ControlEvent>,
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
                if ui.button("Save STL").clicked() {
                    let now = Local::now();
                    events.send(ControlEvent::Save(now));
                    match stl::write_to_stl(&field, now) {
                        Ok(path) => {
                            tracing::info!("Saved STL: {}", path.display());
                        }
                        Err(e) => {
                            tracing::error!("Failed to save STL: {e}");
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
