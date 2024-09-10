use std::sync::Arc;

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use ndarray::{Array2, Zip};
use parking_lot::RwLock;

use crate::{Field, ResetSimulation};

pub struct ReiterSimulatorPlugin;

impl Plugin for ReiterSimulatorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimulationConfig>();
        app.add_systems(Startup, setup);
        app.add_systems(Update, (configure_ui, event_listener));
    }
}

#[derive(Resource, Default)]
struct SimulationConfig(pub Arc<RwLock<SimulationConfigInner>>);

struct SimulationConfigInner {
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
}

impl Default for SimulationConfigInner {
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

pub fn update_grid(s: &mut Array2<f32>, gamma: f32, alpha: f32) {
    let n = s.shape()[0];
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

fn setup(config: Res<SimulationConfig>, field: Res<Field>) {
    let field = Arc::clone(&field.0);
    let config = Arc::clone(&config.0);
    let n = field.read().cells.shape()[0];
    let mut cells = init_grid(n, config.read().beta);

    std::thread::spawn(move || loop {
        let SimulationConfigInner {
            alpha, beta, gamma, ..
        } = *config.read();

        if field.read().step == 0 {
            cells = init_grid(n, beta);
            field.write().cells = cells.mapv(|x| if x >= 1.0 { x } else { 0.0 });
        }
        if !field.read().is_running {
            continue;
        }
        let mut field = field.write();
        field.step += 1;
        update_grid(&mut cells, gamma, alpha);
        field.cells = cells.mapv(|x| if x >= 1.0 { x } else { 0.0 });
    });
}

fn configure_ui(mut contexts: EguiContexts, config: Res<SimulationConfig>) {
    egui::Window::new("Reiter's Snowflake").show(contexts.ctx_mut(), |ui| {
        ui.vertical(|ui| {
            ui.add(egui::Slider::new(&mut config.0.write().alpha, 0.0..=2.0).text("α: diffusion constant"));
            ui.add(egui::Slider::new(&mut config.0.write().beta, 0.0..=1.0).text("β: background field"));
            ui.add(
                egui::Slider::new(&mut config.0.write().gamma, 0.0..=1.0)
                    .text("γ: addition constant")
                    .logarithmic(true),
            );
        });
    });
}

fn event_listener(field: Res<Field>, mut reset_events: EventReader<ResetSimulation>) {
    for _ in reset_events.read() {
        field.0.write().step = 0;
    }
}
