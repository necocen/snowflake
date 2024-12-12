use bevy::prelude::*;

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
struct MeshMaterials(Vec<MeshMaterial2d<ColorMaterial>>);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    field: Res<Field>,
    coordinates: Res<Coordinates>,
) {
    let mut transform = Transform::default();
    transform.rotate_z(30f32.to_radians());
    commands.spawn((Camera2d, transform));
    let n = field.0.read().cells.shape()[0];
    let hexagon = meshes.add(RegularPolygon::new(coordinates.scale / f32::sqrt(3.0), 6));
    let mesh_materials: Vec<MeshMaterial2d<ColorMaterial>> = (0..256)
        .map(|i| {
            let alpha = i as f32 / 255.0;
            MeshMaterial2d(materials.add(ColorMaterial::from(Color::WHITE.with_alpha(alpha))))
        })
        .collect();
    commands.insert_resource(MeshMaterials(mesh_materials.clone()));

    for i in 0..n {
        for j in 0..n {
            let translation = Vec3::new(
                i as f32 + j as f32 / 2.0 - n as f32 * 0.75,
                (j as f32 - (n / 2) as f32) * f32::sqrt(3.0) / 2.0,
                0.0,
            ) * coordinates.scale;
            commands.spawn((
                Cell(i, j, 0),
                Mesh2d(hexagon.clone()),
                mesh_materials[0].clone(),
                Transform::from_translation(translation),
            ));
        }
    }
}

fn update_visualization(
    field: Res<Field>,
    mut query: Query<(
        &mut Cell,
        &mut Visibility,
        &mut MeshMaterial2d<ColorMaterial>,
    )>,
    mesh_materials: Res<MeshMaterials>,
) {
    let new_values = {
        let field = field.0.read();
        let max = field.cells.fold(0.0f32, |a, &b| a.max(b));
        let min = field
            .cells
            .fold(max, |a, &b| if b > 0.0 { a.min(b) } else { a });
        (&field.cells - min) / (max - min)
    };

    for (mut cell, mut visibility, mut mesh_material) in query.iter_mut() {
        let Cell(i, j, value) = &mut *cell;
        let new_value = (new_values[[*i, *j]] * 254.0) as u8; // 0..=254。最終的には1..=255になる。0は透明になってしまうので1から始まるようにする。
        if new_value > 0 {
            if *value == new_value {
                continue;
            }
            *value = new_value;
            let alpha = *value + 1;
            *mesh_material = mesh_materials.0[alpha as usize].clone();
            *visibility = Visibility::Visible;
        } else {
            *visibility = Visibility::Hidden;
        }
    }
}
