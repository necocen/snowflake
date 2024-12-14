use std::sync::Arc;

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use ndarray::Zip;
use parking_lot::RwLock;

use crate::{
    gravner_griffeath::{SimulationConfig, SimulationConfigLog, SimulationConfigLogRecord, State},
    ControlEvent, Field,
};

pub struct GravnerGrifeeathSimulatorMTPlugin;

impl Plugin for GravnerGrifeeathSimulatorMTPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimulationConfigWrapper>();
        app.init_resource::<SimulationConfigLogWrapper>();
        app.add_systems(Startup, setup);
        app.add_systems(Update, (event_listener, configure_ui));
    }
}

#[derive(Resource, Default)]
struct SimulationConfigWrapper(pub Arc<RwLock<SimulationConfig>>);

#[derive(Resource, Default)]
struct SimulationConfigLogWrapper(pub Arc<RwLock<SimulationConfigLog>>);

fn setup(
    config: Res<SimulationConfigWrapper>,
    log: Res<SimulationConfigLogWrapper>,
    field: Res<Field>,
) {
    let field = Arc::clone(&field.0);
    let config = Arc::clone(&config.0);
    let log = Arc::clone(&log.0);
    let n = field.read().cells.shape()[0];
    let mut state = State::new(n, config.read().rho);
    let mut old_config = SimulationConfig::default();

    std::thread::spawn(move || loop {
        let config = *config.read();
        if field.read().step == 0 {
            log.write().clear();
            state = State::new(n, config.rho);
            field.write().cells =
                Zip::from(&state.a)
                    .and(&state.c)
                    .par_map_collect(|&a, &c| if a { c } else { 0.0 });
        }
        if !field.read().is_running {
            continue;
        }
        if old_config != config || field.read().step == 0 {
            tracing::info!("Step: {}, {config:?}", field.read().step);
            log.write()
                .push(SimulationConfigLogRecord::new(field.read().step, &config));
            old_config = config;
        }
        let mut field = field.write();
        if field.step % 100 == 0 {
            let total_mass = state.b.sum() + state.c.sum() + state.d.sum();
            tracing::debug!("step: {}, total_mass: {total_mass}", field.step);
        }
        field.step += 1;
        state.update(config);
        field.cells = Zip::from(&state.a)
            .and(&state.c)
            .par_map_collect(|&a, &c| if a { c } else { 0.0 });
    });
}

fn event_listener(
    field: Res<Field>,
    log: Res<SimulationConfigLogWrapper>,
    mut reset_events: EventReader<ControlEvent>,
) {
    for event in reset_events.read() {
        match event {
            ControlEvent::Reset => {
                field.0.write().step = 0;
            }
            ControlEvent::Save(now) => match log.0.read().save_to_csv(*now) {
                Ok(path) => {
                    tracing::info!("Saved CSV: {}", path.display());
                }
                Err(e) => {
                    tracing::error!("Failed to save CSV: {e}");
                }
            },
        }
    }
}

fn configure_ui(mut contexts: EguiContexts, config: Res<SimulationConfigWrapper>) {
    egui::Window::new("Gravner-Griffeath's Snowflake").show(contexts.ctx_mut(), |ui| {
        ui.vertical(|ui| {
            ui.add(
                egui::Slider::new(&mut config.0.write().rho, 0.0..=1.0).text("ρ: vapor density"),
            );
            ui.add(egui::Slider::new(&mut config.0.write().beta, 1.0..=4.0).text("β: anisotropy"));
            ui.add(
                egui::Slider::new(&mut config.0.write().alpha, 0.0..=1.0)
                    .text("α: attachment threshold for b"),
            );
            ui.add(
                egui::Slider::new(&mut config.0.write().theta, 0.0..=0.5)
                    .text("θ: attachment threshold for d")
                    .logarithmic(true),
            );
            ui.add(
                egui::Slider::new(&mut config.0.write().kappa, 0.0..=1.0)
                    .text("κ: freezing rate")
                    .logarithmic(true),
            );
            ui.add(egui::Slider::new(&mut config.0.write().mu, 0.0..=0.3).text("μ: melting rate"));
            ui.add(
                egui::Slider::new(&mut config.0.write().gamma, 0.0..=0.01)
                    .text("γ: sublimation rate")
                    .logarithmic(true),
            );
            ui.add(
                egui::Slider::new(&mut config.0.write().sigma, 0.0..=1.0)
                    .text("σ: noise")
                    .logarithmic(true),
            );
        });
    });
}
