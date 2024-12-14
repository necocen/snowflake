use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use ndarray::{Array2, Zip};

use crate::{ControlEvent, Field};

pub struct ReiterSimulatorPlugin;

impl Plugin for ReiterSimulatorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimulationConfig>();
        app.init_resource::<State>();
        app.add_systems(Update, (configure_ui, event_listener, update_simulation));
    }
}

#[derive(Default, Resource)]
struct State {
    pub s: Array2<f32>,
}

impl State {
    pub fn new(n: usize, beta: f32) -> Self {
        let mut s = Array2::<f32>::ones((n, n)) * beta;
        s[[n / 2, n / 2]] = 1.0;
        Self { s }
    }

    pub fn update(&mut self, config: SimulationConfig) {
        let n = self.s.shape()[0];
        let s_view = self.s.view();
        // receptive_cells と uv0 を同時に計算
        let combined = Zip::indexed(self.s.view()).par_map_collect(|(i, j), &s_val| {
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
        Zip::indexed(&mut self.s).and(&combined).par_for_each(
            |(i, j), s_val, &(receptive, (u0, v0))| {
                let mut v1 = v0;
                // Rule 1
                if receptive {
                    v1 += config.gamma;
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

                let u1 = u0 + config.alpha * (u0_neighbors - u0) / 2.0;
                *s_val = u1 + v1;
            },
        );
    }
}

#[derive(Debug, Clone, Copy, Resource)]
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

fn update_simulation(
    mut field: ResMut<Field>,
    mut state: ResMut<State>,
    config: Res<SimulationConfig>,
) {
    let n = field.cells.shape()[0];
    if field.step == 0 {
        *state = State::new(n, config.beta);
        field.cells = state.s.mapv(|x| if x >= 1.0 { x } else { 0.0 });
    }
    if !field.is_running {
        return;
    }
    if field.step % 100 == 0 {
        tracing::debug!("step: {}", field.step);
    }
    field.step += 1;
    state.update(*config);
    field.cells = state.s.mapv(|x| if x >= 1.0 { x } else { 0.0 });
}

fn configure_ui(mut contexts: EguiContexts, mut config: ResMut<SimulationConfig>) {
    egui::Window::new("Reiter's Snowflake").show(contexts.ctx_mut(), |ui| {
        ui.vertical(|ui| {
            ui.add(egui::Slider::new(&mut config.alpha, 0.0..=2.0).text("α: diffusion constant"));
            ui.add(egui::Slider::new(&mut config.beta, 0.0..=1.0).text("β: background field"));
            ui.add(
                egui::Slider::new(&mut config.gamma, 0.0..=1.0)
                    .text("γ: addition constant")
                    .logarithmic(true),
            );
        });
    });
}

fn event_listener(mut field: ResMut<Field>, mut reset_events: EventReader<ControlEvent>) {
    for event in reset_events.read() {
        if let ControlEvent::Reset = event {
            field.step = 0;
        }
    }
}
