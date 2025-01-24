use bevy::{prelude::*, render::view::NoFrustumCulling};

use crate::{
    instancing::{CustomMaterialPlugin, InstanceData, InstanceMaterialData},
    Field,
};

#[derive(Resource)]
struct Coordinates {
    scale: f32,
}

impl Default for Coordinates {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

pub struct VisualizationPlugin;

impl Plugin for VisualizationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Coordinates>();
        app.add_plugins(CustomMaterialPlugin);
        app.add_systems(Startup, setup);
        app.add_systems(Update, update_visualization);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    field: Res<Field>,
    coordinates: Res<Coordinates>,
) {
    let n = field.cells.shape()[0];

    let hexagon = meshes.add(RegularPolygon::new(coordinates.scale / f32::sqrt(3.0), 6));
    let instance_data = (0..n)
        .flat_map(|i| {
            (0..n).map(move |j| {
                Vec3::new(
                    i as f32 + j as f32 / 2.0 - n as f32 * 0.75,
                    (j as f32 - (n / 2) as f32) * f32::sqrt(3.0) / 2.0,
                    0.0,
                )
            })
        })
        .map(|position| InstanceData {
            position: position * coordinates.scale,
            scale: 1.0,
            color: [1.0, 1.0, 1.0, 0.0],
        })
        .collect::<Vec<_>>();

    let mut transform = Transform::default();
    transform.rotate_z(30f32.to_radians());
    commands.spawn((
        Mesh2d(hexagon),
        InstanceMaterialData(instance_data),
        NoFrustumCulling,
    ));
    commands.spawn((Camera2d, transform));
}

fn update_visualization(field: Res<Field>, mut query: Query<&mut InstanceMaterialData>) {
    let new_values = {
        let max = field.cells.fold(0.0f32, |a, &b| a.max(b));
        &field.cells / max
    };
    let n = field.cells.shape()[0];
    query.iter_mut().for_each(|mut instance_data| {
        instance_data
            .0
            .iter_mut()
            .enumerate()
            .for_each(|(i, data)| {
                data.color[3] = new_values[[i / n, i % n]];
            });
    });
}
