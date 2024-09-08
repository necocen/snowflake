use std::sync::Arc;

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use ndarray::{parallel::prelude::*, Array2, Zip};
use parking_lot::RwLock;

use crate::{Field, ResetSimulation};

pub struct GravnerGrifeeathSimulatorPlugin;

impl Plugin for GravnerGrifeeathSimulatorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimulationConfig>();
        app.add_systems(Startup, setup);
        app.add_systems(Update, event_listener);
    }
}

#[derive(Resource, Default)]
struct SimulationConfig(pub Arc<RwLock<SimulationConfigInner>>);

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
    let n = field.read().frozen_cells.shape()[0];
    let mut state = State::new(n, config.read().rho);

    std::thread::spawn(move || loop {
        let SimulationConfigInner {
            rho,
            beta,
            alpha,
            theta,
            kappa,
            mu,
            gamma,
            sigma,
        } = *config.read();

        if field.read().step == 0 {
            state = State::new(n, rho);
            field.write().frozen_cells = state.a.clone();
        }
        if !field.read().is_running {
            continue;
        }
        let mut field = field.write();
        if field.step % 100 == 0 {
            let total_mass = state.b.sum() + state.c.sum() + state.d.sum();
            tracing::info!("step: {}, total_mass: {total_mass}", field.step);
        }
        field.step += 1;
        state.update(beta, alpha, theta, kappa, mu, gamma, sigma);
        field.frozen_cells = state.a.clone();
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
        let a = self.a.view();
        let snowflake = a.mapv(|v| v as u8 as f32);
        let neighbors = Zip::indexed(&a).par_map_collect(|(i, j), _| {
            a[[(i + 1) % n, j]] as u8
                + a[[(i + n - 1) % n, j]] as u8
                + a[[i, (j + 1) % n]] as u8
                + a[[i, (j + n - 1) % n]] as u8
                + a[[(i + n - 1) % n, (j + 1) % n]] as u8
                + a[[(i + 1) % n, (j + n - 1) % n]] as u8
        });
        let boundary = neighbors.mapv(|n| (n > 0) as u8 as f32) * (1.0 - &snowflake);

        // (i) Diffusion
        let d = self.d.view();
        let d = (Zip::indexed(&d).par_map_collect(|(i, j), &d_val| {
            d_val
                + d[[(i + 1) % n, j]]
                + d[[(i + n - 1) % n, j]]
                + d[[i, (j + 1) % n]]
                + d[[i, (j + n - 1) % n]]
                + d[[(i + n - 1) % n, (j + 1) % n]]
                + d[[(i + 1) % n, (j + n - 1) % n]]
        }) + neighbors.mapv(|v| v as f32) * &d)
            / 7.0
            * (1.0 - &snowflake);

        // (ii) Freezing
        let db = &d * &boundary;
        let b = &self.b + (1.0 - kappa) * &db;
        let c = &self.c + kappa * &db;
        let d = d - &db;

        // (iii) Attachment
        let a = Zip::indexed(&a).par_map_collect(|(i, j), &a| {
            if boundary[[i, j]] == 0.0 {
                return a;
            }
            match neighbors[[i, j]] {
                // not a boundary cell
                0 => panic!("not a boundary cell"),
                // tip or flat spot
                1..=2 => b[[i, j]] >= beta,
                // concave spot
                3 => {
                    // b(x) >= 1.0 or [b(x) >= alpha and Σ_{y: neighbor of x} d(y) < theta]
                    b[[i, j]] >= 1.0
                        || (b[[i, j]] >= alpha
                            && (d[[(i + 1) % n, j]]
                                + d[[(i + n - 1) % n, j]]
                                + d[[i, (j + 1) % n]]
                                + d[[i, (j + n - 1) % n]]
                                + d[[(i + n - 1) % n, (j + 1) % n]]
                                + d[[(i + 1) % n, (j + n - 1) % n]])
                                < theta)
                }
                _ => true,
            }
        });

        let ab = Zip::from(&a)
            .and(&b)
            .par_map_collect(|&a, &b| if a { b } else { 0.0 });
        let b = b - &ab;
        let c = c + &ab;

        // (iv) Melting
        let boundary = Zip::indexed(&a).par_map_collect(|(i, j), a_val| {
            (!a_val
                && (a[[(i + 1) % n, j]]
                    || a[[(i + n - 1) % n, j]]
                    || a[[i, (j + 1) % n]]
                    || a[[i, (j + n - 1) % n]]
                    || a[[(i + n - 1) % n, (j + 1) % n]]
                    || a[[(i + 1) % n, (j + n - 1) % n]])) as u8 as f32
        });
        let mbb = mu * &b * &boundary;
        let gcb = gamma * &c * &boundary;
        let b = b - &mbb;
        let c = c - &gcb;
        let d = d + &mbb + &gcb;

        // (v) Noise
        // TODO

        self.a = a;
        self.b = b;
        self.c = c;
        self.d = d;
    }
}

fn event_listener(field: Res<Field>, mut reset_events: EventReader<ResetSimulation>) {
    for _ in reset_events.read() {
        field.0.write().step = 0;
    }
}
