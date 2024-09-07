use bevy::{
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use fnv::FnvHashMap;

use crate::Field;

pub struct VisualizationPlugin;

impl Plugin for VisualizationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Coordinates>();
        app.add_systems(Startup, setup);
        app.add_systems(Update, update_visualization);
    }
}

#[derive(Resource)]
struct Coordinates {
    entities: FnvHashMap<(usize, usize), Entity>,
    scale: f32,
}

impl Default for Coordinates {
    fn default() -> Self {
        Self {
            entities: FnvHashMap::default(),
            scale: 1.0,
        }
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    field: Res<Field>,
    mut coordinates: ResMut<Coordinates>,
) {
    commands.spawn(Camera2dBundle::default());
    let n = field.0.read().frozen_cells.shape()[0];

    let hexagon =
        Mesh2dHandle(meshes.add(RegularPolygon::new(coordinates.scale / f32::sqrt(3.0), 6)));
    let color = materials.add(Color::WHITE);

    for i in 0..n {
        for j in 0..n {
            if let Some(id) = coordinates.entities.get(&(i, j)) {
                commands.entity(*id).despawn();
            }
            let translation = Vec3::new(
                i as f32 + j as f32 / 2.0 - n as f32 * 0.75,
                (j as f32 - (n / 2) as f32) * f32::sqrt(3.0) / 2.0,
                0.0,
            ) * coordinates.scale;
            let id = commands
                .spawn(MaterialMesh2dBundle {
                    visibility: Visibility::Hidden,
                    mesh: hexagon.clone(),
                    material: color.clone(),
                    transform: Transform::from_translation(translation),
                    ..default()
                })
                .id();
            coordinates.entities.insert((i, j), id);
        }
    }
}

fn update_visualization(mut commands: Commands, field: Res<Field>, coordinates: Res<Coordinates>) {
    let field = field.0.read();
    let n = field.frozen_cells.shape()[0];
    for i in 0..n {
        for j in 0..n {
            let id = coordinates.entities.get(&(i, j)).unwrap();
            if field.frozen_cells[[i, j]] {
                commands.entity(*id).insert(Visibility::Visible);
            } else {
                commands.entity(*id).insert(Visibility::Hidden);
            }
        }
    }
}
