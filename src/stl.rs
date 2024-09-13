use bevy::math::Vec3;
use fnv::{FnvHashMap, FnvHashSet};
use ndarray::Array2;
use stl_io::{Normal, Triangle, Vertex};

#[derive(Clone, Copy)]
pub struct Facet(Vec3, Vec3, Vec3);

pub fn cells_to_facets(cells: &Array2<f32>, xy_scale: f32, z_scale: f32) -> Vec<Facet> {
    let n = cells.shape()[0];
    let x_offset = (n as f32 - 1.0) * xy_scale / 2.0;
    let y_offset = (n as f32 - 1.0) * xy_scale / 3.0f32.sqrt();
    let mut facets = Vec::new();
    let mut segments: FnvHashMap<(usize, usize), FnvHashSet<(usize, usize)>> =
        FnvHashMap::default();
    let directions0 = [(0, 0), (1, 0), (0, 1)];
    let directions1 = [(1, 0), (1, 1), (0, 1)];
    let sqrt3_2 = 3.0f32.sqrt() / 2.0;
    for i in 0..(n - 1) {
        for j in 0..(n - 1) {
            let i1 = i + 1;
            let j1 = j + 1;
            /*
                p10 <-- p11
                /   \   /
              p00 --> p01
            */
            let p00 = Vec3::new(
                (i as f32 + j as f32 * 0.5) * xy_scale - x_offset,
                j as f32 * sqrt3_2 * xy_scale - y_offset,
                cells[[i, j]] * z_scale,
            );
            let p01 = Vec3::new(
                (i1 as f32 + j as f32 * 0.5) * xy_scale - x_offset,
                j as f32 * sqrt3_2 * xy_scale - y_offset,
                cells[[i1, j]] * z_scale,
            );
            let p10 = Vec3::new(
                (i as f32 + j1 as f32 * 0.5) * xy_scale - x_offset,
                j1 as f32 * sqrt3_2 * xy_scale - y_offset,
                cells[[i, j1]] * z_scale,
            );
            let p11 = Vec3::new(
                (i1 as f32 + j1 as f32 * 0.5) * xy_scale - x_offset,
                j1 as f32 * sqrt3_2 * xy_scale - y_offset,
                cells[[i1, j1]] * z_scale,
            );
            if cells[[i, j]] > 0.0 && cells[[i1, j]] > 0.0 && cells[[i, j1]] > 0.0 {
                facets.push(Facet(p00, p01, p10));
                facets.push(Facet(
                    p10.with_z(-p10.z),
                    p01.with_z(-p01.z),
                    p00.with_z(-p00.z),
                ));
            }

            if cells[[i1, j]] > 0.0 && cells[[i1, j1]] > 0.0 && cells[[i, j1]] > 0.0 {
                facets.push(Facet(p01, p11, p10));
                facets.push(Facet(
                    p10.with_z(-p10.z),
                    p11.with_z(-p11.z),
                    p01.with_z(-p01.z),
                ));
            }

            if cells[[i, j]] > 0.0 && cells[[i1, j]] > 0.0 && cells[[i, j1]] > 0.0 {
                for k in 0..3 {
                    let start = (i + directions0[k].0, j + directions0[k].1);
                    let end = (
                        i + directions0[(k + 1) % 3].0,
                        j + directions0[(k + 1) % 3].1,
                    );
                    if let Some(end_set) = segments.get_mut(&end) {
                        if !end_set.remove(&start) {
                            segments.entry(start).or_default().insert(end);
                        } else if end_set.is_empty() {
                            segments.remove(&end);
                        }
                    } else {
                        segments.entry(start).or_default().insert(end);
                    }
                }
            }
            if cells[[i1, j]] > 0.0 && cells[[i1, j1]] > 0.0 && cells[[i, j1]] > 0.0 {
                for k in 0..3 {
                    let start = (i + directions1[k].0, j + directions1[k].1);
                    let end = (
                        i + directions1[(k + 1) % 3].0,
                        j + directions1[(k + 1) % 3].1,
                    );
                    if let Some(end_set) = segments.get_mut(&end) {
                        if !end_set.remove(&start) {
                            segments.entry(start).or_default().insert(end);
                        } else if end_set.is_empty() {
                            segments.remove(&end);
                        }
                    } else {
                        segments.entry(start).or_default().insert(end);
                    }
                }
            }
        }
    }

    let mut contours = Vec::new();
    while !segments.is_empty() {
        let start = *segments.keys().next().unwrap();
        let mut contour = vec![start];
        let mut current = start;
        while let Some(next_set) = segments.get_mut(&current) {
            if next_set.len() != 1 {
                tracing::warn!("Should be 1");
            }
            let next = *next_set.iter().next().unwrap();
            contour.push(next);
            segments.remove(&current);
            current = next;
        }
        contours.push(contour);
    }

    for contour in contours {
        for i in 0..(contour.len() - 1) {
            let c0x = contour[i].0 as f32 + contour[i].1 as f32 * 0.5;
            let c0y = contour[i].1 as f32 * sqrt3_2;
            let c1x = contour[i + 1].0 as f32 + contour[i + 1].1 as f32 * 0.5;
            let c1y = contour[i + 1].1 as f32 * sqrt3_2;
            let p00 = Vec3::new(
                c0x * xy_scale - x_offset,
                c0y * xy_scale - y_offset,
                cells[[contour[i].0, contour[i].1]] * z_scale,
            );
            let p01 = Vec3::new(
                c1x * xy_scale - x_offset,
                c1y * xy_scale - y_offset,
                cells[[contour[i + 1].0, contour[i + 1].1]] * z_scale,
            );
            facets.push(Facet(p01, p00, p01.with_z(-p01.z)));
            facets.push(Facet(p00.with_z(-p00.z), p01.with_z(-p01.z), p00));
        }
    }

    facets
}

pub fn calculate_normal(facets: Vec<Facet>) -> Vec<Triangle> {
    facets
        .into_iter()
        .map(|Facet(p0, p1, p2)| {
            let normal = (p1 - p0).cross(p2 - p1).normalize();
            Triangle {
                vertices: [
                    Vertex::new([p0.x, p0.y, p0.z]),
                    Vertex::new([p1.x, p1.y, p1.z]),
                    Vertex::new([p2.x, p2.y, p2.z]),
                ],
                normal: Normal::new([normal.x, normal.y, normal.z]),
            }
        })
        .collect()
}
