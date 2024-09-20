use bevy::{prelude::*, sprite::Mesh2dHandle};

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
    scale: f32,
}

impl Default for Coordinates {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

#[derive(Component)]
struct Cell(usize, usize, u8);

#[derive(Resource)]
struct MaterialHandles {
    handles: Vec<Handle<ColorMaterial>>,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    field: Res<Field>,
    coordinates: Res<Coordinates>,
) {
    let mut transform = Transform::default();
    transform.rotate_z(30f32.to_radians());
    commands.spawn(Camera2dBundle {
        transform,
        ..default()
    });
    let n = field.0.read().cells.shape()[0];
    let hexagon =
        Mesh2dHandle(meshes.add(RegularPolygon::new(coordinates.scale / f32::sqrt(3.0), 6)));
    let material_handles: Vec<Handle<ColorMaterial>> = (0..256)
        .map(|i| {
            let alpha = i as f32 / 255.0;
            materials.add(ColorMaterial::from(Color::WHITE.with_alpha(alpha)))
        })
        .collect();
    commands.insert_resource(MaterialHandles {
        handles: material_handles.clone(),
    });

    for i in 0..n {
        for j in 0..n {
            let translation = Vec3::new(
                i as f32 + j as f32 / 2.0 - n as f32 * 0.75,
                (j as f32 - (n / 2) as f32) * f32::sqrt(3.0) / 2.0,
                0.0,
            ) * coordinates.scale;
            commands.spawn((
                Cell(i, j, 0),
                ColorMesh2dBundle {
                    visibility: Visibility::Hidden,
                    mesh: hexagon.clone(),
                    material: material_handles[0].clone(),
                    transform: Transform::from_translation(translation),
                    ..default()
                },
            ));
        }
    }
}

fn update_visualization(
    field: Res<Field>,
    mut query: Query<(&mut Cell, &mut Visibility, &mut Handle<ColorMaterial>)>,
    material_handles: Res<MaterialHandles>,
) {
    let new_values = {
        let field = field.0.read();
        let max = field.cells.fold(0.0f32, |a, &b| a.max(b));
        let min = field
            .cells
            .fold(max, |a, &b| if b > 0.0 { a.min(b) } else { a });
        (&field.cells - min) / (max - min)
    };

    for (mut cell, mut visibility, mut material_handle) in query.iter_mut() {
        let Cell(i, j, value) = &mut *cell;
        let new_value = (new_values[[*i, *j]] * 254.0) as u8; // 0..=254。最終的には1..=255になる。0は透明になってしまうので1から始まるようにする。
        if new_value > 0 {
            if *value == new_value {
                continue;
            }
            *value = new_value;
            let alpha = 255 - *value;
            *material_handle = material_handles.handles[alpha as usize].clone();
            *visibility = Visibility::Visible;
        } else {
            *visibility = Visibility::Hidden;
        }
    }
}
