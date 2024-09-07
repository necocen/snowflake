use bevy::prelude::*;
use bevy_egui::EguiPlugin;

mod reiter;
mod utils;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, EguiPlugin))
        .add_plugins(reiter::ReiterSimulatorPlugin)
        .run();
}
