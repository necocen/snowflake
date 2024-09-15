use std::sync::Arc;

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use ndarray::{Array2, Zip};
use ndarray_rand::{rand_distr::Standard, RandomExt as _};
use parking_lot::RwLock;

use crate::{Field, ResetSimulation};

pub struct GravnerGrifeeathSimulatorPlugin;

impl Plugin for GravnerGrifeeathSimulatorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimulationConfig>();
        app.add_systems(Startup, setup);
        app.add_systems(Update, (event_listener, configure_ui));
    }
}

#[derive(Resource, Default)]
struct SimulationConfig(pub Arc<RwLock<SimulationConfigInner>>);

#[derive(Debug, Clone, Copy, PartialEq)]
struct SimulationConfigInner {
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

impl Default for SimulationConfigInner {
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

fn setup(config: Res<SimulationConfig>, field: Res<Field>) {
    let field = Arc::clone(&field.0);
    let config = Arc::clone(&config.0);
    let n = field.read().cells.shape()[0];
    let mut state = State::new(n, config.read().rho);
    let mut old_config = SimulationConfigInner::default();

    std::thread::spawn(move || loop {
        let config = *config.read();
        let SimulationConfigInner {
            rho,
            beta,
            alpha,
            theta,
            kappa,
            mu,
            gamma,
            sigma,
        } = config;
        if field.read().step == 0 {
            state = State::new(n, rho);
            field.write().cells =
                Zip::from(&state.a)
                    .and(&state.c)
                    .par_map_collect(|&a, &c| if a { c } else { 0.0 });
        }
        if !field.read().is_running {
            continue;
        }
        if old_config != config {
            tracing::info!("Step: {}, {config:?}", field.read().step);
            old_config = config;
        }
        let mut field = field.write();
        if field.step % 100 == 0 {
            let total_mass = state.b.sum() + state.c.sum() + state.d.sum();
            tracing::debug!("step: {}, total_mass: {total_mass}", field.step);
        }
        field.step += 1;
        state.update(beta, alpha, theta, kappa, mu, gamma, sigma);
        field.cells = Zip::from(&state.a)
            .and(&state.c)
            .par_map_collect(|&a, &c| if a { c } else { 0.0 });
    });
}

struct State {
    a: Array2<bool>,
    b: Array2<f32>,
    c: Array2<f32>,
    d: Array2<f32>,
}

impl State {
    fn new(n: usize, rho: f32) -> Self {
        let mut a = Array2::<bool>::default((n, n));
        a[[n / 2, n / 2]] = true;

        let b = Array2::<f32>::zeros((n, n));

        let mut c = Array2::<f32>::zeros((n, n));
        c[[n / 2, n / 2]] = 1.0;

        let mut d = Array2::<f32>::ones((n, n)) * rho;
        d[[n / 2, n / 2]] = 0.0;

        Self { a, b, c, d }
    }

    fn update(
        &mut self,
        beta: f32,
        alpha: f32,
        theta: f32,
        kappa: f32,
        mu: f32,
        gamma: f32,
        sigma: f32,
    ) {
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

fn event_listener(field: Res<Field>, mut reset_events: EventReader<ResetSimulation>) {
    for _ in reset_events.read() {
        field.0.write().step = 0;
    }
}

fn configure_ui(mut contexts: EguiContexts, config: Res<SimulationConfig>) {
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
