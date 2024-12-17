use std::{fs::OpenOptions, path::PathBuf};

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use chrono::{DateTime, Local};
use ndarray::{Array2, Zip};
use ndarray_rand::{rand_distr::Standard, RandomExt as _};

use crate::{ControlEvent, Field};

pub struct GravnerGriffeathSimulatorPlugin;

impl Plugin for GravnerGriffeathSimulatorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimulationConfig>();
        app.init_resource::<SimulationConfigLog>();
        app.init_resource::<State>();
        app.add_systems(Main, update_simulation);
        app.add_systems(Update, (event_listener, configure_ui));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Resource)]
pub struct SimulationConfig {
    /// vapor density parameter
    pub rho: f32,
    /// tip attachment threshold for b (anisotropy parameter)
    pub beta: f32,
    /// concave attachment threshold for b
    pub alpha: f32,
    /// concave attachment threshold for d
    pub theta: f32,
    /// crystalization parameter
    pub kappa: f32,
    /// melting parameter
    pub mu: f32,
    /// sublimation parameter
    pub gamma: f32,
    /// perturbation strength
    pub sigma: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            rho: 0.5,
            beta: 1.4,
            alpha: 0.1,
            theta: 0.005,
            kappa: 0.001,
            mu: 0.06,
            gamma: 0.001,
            sigma: 0.0,
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct SimulationConfigLogRecord {
    pub step: u64,
    #[serde(rename = "ρ")]
    pub rho: f32,
    #[serde(rename = "β")]
    pub beta: f32,
    #[serde(rename = "α")]
    pub alpha: f32,
    #[serde(rename = "θ")]
    pub theta: f32,
    #[serde(rename = "κ")]
    pub kappa: f32,
    #[serde(rename = "μ")]
    pub mu: f32,
    #[serde(rename = "γ")]
    pub gamma: f32,
    #[serde(rename = "σ")]
    pub sigma: f32,
}

impl SimulationConfigLogRecord {
    pub fn new(step: u64, config: &SimulationConfig) -> Self {
        Self {
            step,
            rho: config.rho,
            beta: config.beta,
            alpha: config.alpha,
            theta: config.theta,
            kappa: config.kappa,
            mu: config.mu,
            gamma: config.gamma,
            sigma: config.sigma,
        }
    }
}

#[derive(Default, Resource)]
pub struct SimulationConfigLog {
    log: Vec<SimulationConfigLogRecord>,
}

impl SimulationConfigLog {
    pub fn save_to_csv(&self, now: DateTime<Local>) -> std::io::Result<PathBuf> {
        let filename = format!("snowflake-{}.csv", now.format("%Y%m%d%H%M%S"));
        let path = PathBuf::from(&filename);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let mut writer = csv::Writer::from_writer(file);
        for record in &self.log {
            writer.serialize(record)?;
        }
        writer.flush()?;
        Ok(path)
    }

    pub fn push(&mut self, record: SimulationConfigLogRecord) {
        self.log.push(record);
    }

    pub fn last(&self) -> Option<SimulationConfig> {
        let config = self.log.last()?;
        Some(SimulationConfig {
            rho: config.rho,
            beta: config.beta,
            alpha: config.alpha,
            theta: config.theta,
            kappa: config.kappa,
            mu: config.mu,
            gamma: config.gamma,
            sigma: config.sigma,
        })
    }

    pub fn clear(&mut self) {
        self.log.clear();
    }
}

#[derive(Default, Resource)]
pub struct State {
    pub a: Array2<bool>,
    pub b: Array2<f32>,
    pub c: Array2<f32>,
    pub d: Array2<f32>,
}

impl State {
    pub fn new(n: usize, rho: f32) -> Self {
        let center = [n / 2, n / 2];
        let mut a = Array2::<bool>::default((n, n));
        a[center] = true;
        let b = Array2::<f32>::zeros((n, n));
        let mut c = Array2::<f32>::zeros((n, n));
        c[center] = 1.0;
        let mut d = Array2::<f32>::ones((n, n)) * rho;
        d[center] = 0.0;

        Self { a, b, c, d }
    }

    pub fn update(&mut self, config: SimulationConfig) {
        let SimulationConfig {
            beta,
            alpha,
            theta,
            kappa,
            mu,
            gamma,
            sigma,
            ..
        } = config;
        let n = self.a.shape()[0];

        let neighbors = Zip::indexed(&self.a).par_map_collect(|(i, j), _| {
            self.a[[(i + 1) % n, j]] as u8
                + self.a[[(i + n - 1) % n, j]] as u8
                + self.a[[i, (j + 1) % n]] as u8
                + self.a[[i, (j + n - 1) % n]] as u8
                + self.a[[(i + n - 1) % n, (j + 1) % n]] as u8
                + self.a[[(i + 1) % n, (j + n - 1) % n]] as u8
        });

        // (i) Diffusion
        let mut d_new = Array2::<f32>::zeros(self.d.raw_dim());
        Zip::indexed(&mut d_new)
            .and(&self.a)
            .and(&self.d)
            .and(&neighbors)
            .par_for_each(|(i, j), d, &a_old, &d_old, &neighbors| {
                if !a_old {
                    *d = (d_old
                        + self.d[[(i + 1) % n, j]]
                        + self.d[[(i + n - 1) % n, j]]
                        + self.d[[i, (j + 1) % n]]
                        + self.d[[i, (j + n - 1) % n]]
                        + self.d[[(i + n - 1) % n, (j + 1) % n]]
                        + self.d[[(i + 1) % n, (j + n - 1) % n]]
                        + neighbors as f32 * d_old)
                        / 7.0;
                }
            });

        // (ii) Freezing
        let mut b_new = self.b.clone();
        let mut c_new = self.c.clone();
        Zip::from(&self.a)
            .and(&mut b_new)
            .and(&mut c_new)
            .and(&mut d_new)
            .and(&neighbors)
            .par_for_each(|&a, b, c, d, &neighbors| {
                if !a && neighbors > 0 {
                    *b += (1.0 - kappa) * *d;
                    *c += kappa * *d;
                    *d = 0.0;
                }
            });

        // (iii) Attachment
        let mut a_new = self.a.clone();
        Zip::indexed(&mut a_new)
            .and(&mut b_new)
            .and(&mut c_new)
            .and(&self.a)
            .and(&neighbors)
            .par_for_each(|(i, j), a, b, c, &a_old, &neighbors| {
                if a_old || neighbors == 0 {
                    return;
                }

                *a = match neighbors {
                    0 => panic!("not a boundary cell"),
                    1..=2 => *b >= beta,
                    3 => {
                        // b(x) >= 1.0 or [b(x) >= alpha and Σ_{y: neighbor of x} d(y) < theta]
                        *b >= 1.0
                            || (*b >= alpha
                                && d_new[[(i + 1) % n, j]]
                                    + d_new[[(i + n - 1) % n, j]]
                                    + d_new[[i, (j + 1) % n]]
                                    + d_new[[i, (j + n - 1) % n]]
                                    + d_new[[(i + n - 1) % n, (j + 1) % n]]
                                    + d_new[[(i + 1) % n, (j + n - 1) % n]]
                                    < theta)
                    }
                    _ => true,
                };

                if *a {
                    *c += *b;
                    *b = 0.0;
                }
            });

        // (iv) Melting
        Zip::indexed(&mut b_new)
            .and(&mut c_new)
            .and(&mut d_new)
            .par_for_each(|(i, j), b, c, d| {
                let boundary = !a_new[[i, j]]
                    && (a_new[[(i + 1) % n, j]]
                        || a_new[[(i + n - 1) % n, j]]
                        || a_new[[i, (j + 1) % n]]
                        || a_new[[i, (j + n - 1) % n]]
                        || a_new[[(i + n - 1) % n, (j + 1) % n]]
                        || a_new[[(i + 1) % n, (j + n - 1) % n]]);
                if boundary {
                    let mu_b = mu * *b;
                    let gamma_c = gamma * *c;
                    *b -= mu_b;
                    *c -= gamma_c;
                    *d += mu_b + gamma_c;
                }
            });

        // (v) Noise
        if sigma.abs() > 0.0 {
            let noise = Array2::<bool>::random(d_new.raw_dim(), Standard);
            Zip::from(&mut d_new).and(&noise).par_for_each(|d, &noise| {
                if noise {
                    *d *= 1.0 + sigma;
                } else {
                    *d *= 1.0 - sigma;
                }
            });
        }

        self.a = a_new;
        self.b = b_new;
        self.c = c_new;
        self.d = d_new;
    }
}

fn event_listener(
    mut field: ResMut<Field>,
    log: Res<SimulationConfigLog>,
    mut reset_events: EventReader<ControlEvent>,
) {
    for event in reset_events.read() {
        match event {
            ControlEvent::Reset => {
                field.step = 0;
            }
            ControlEvent::Save(now) => match log.save_to_csv(*now) {
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

fn update_simulation(
    mut field: ResMut<Field>,
    mut state: ResMut<State>,
    config: Res<SimulationConfig>,
    mut log: ResMut<SimulationConfigLog>,
) {
    let n = field.cells.shape()[0];
    if field.step == 0 {
        log.clear();
        *state = State::new(n, config.rho);
        field.cells = Zip::from(&state.a)
            .and(&state.c)
            .map_collect(|&a, &c| if a { c } else { 0.0 });
    }
    if !field.is_running {
        return;
    }
    let old_config = log.last().unwrap_or_default();
    if old_config != *config || field.step == 0 {
        tracing::info!("Step: {}, {config:?}", field.step);
        log.push(SimulationConfigLogRecord::new(field.step, &config));
    }

    for _ in 0..5 {
        if field.step % 100 == 0 {
            let total_mass = state.b.sum() + state.c.sum() + state.d.sum();
            tracing::debug!("step: {}, total_mass: {total_mass}", field.step);
        }
        field.step += 1;
        state.update(*config);
    }

    field.cells = Zip::from(&state.a)
        .and(&state.c)
        .map_collect(|&a, &c| if a { c } else { 0.0 });
}

fn configure_ui(mut contexts: EguiContexts, mut config: ResMut<SimulationConfig>) {
    egui::Window::new("Gravner-Griffeath's Snowflake").show(contexts.ctx_mut(), |ui| {
        ui.vertical(|ui| {
            ui.add(egui::Slider::new(&mut config.rho, 0.0..=1.0).text("ρ: vapor density"));
            ui.add(egui::Slider::new(&mut config.beta, 1.0..=4.0).text("β: anisotropy"));
            ui.add(
                egui::Slider::new(&mut config.alpha, 0.0..=1.0)
                    .text("α: attachment threshold for b"),
            );
            ui.add(
                egui::Slider::new(&mut config.theta, 0.0..=0.5)
                    .text("θ: attachment threshold for d")
                    .logarithmic(true),
            );
            ui.add(
                egui::Slider::new(&mut config.kappa, 0.0..=1.0)
                    .text("κ: freezing rate")
                    .logarithmic(true),
            );
            ui.add(egui::Slider::new(&mut config.mu, 0.0..=0.3).text("μ: melting rate"));
            ui.add(
                egui::Slider::new(&mut config.gamma, 0.0..=0.01)
                    .text("γ: sublimation rate")
                    .logarithmic(true),
            );
            ui.add(
                egui::Slider::new(&mut config.sigma, 0.0..=1.0)
                    .text("σ: noise")
                    .logarithmic(true),
            );
        });
    });
}
