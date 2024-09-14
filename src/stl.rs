use std::{fs::OpenOptions, path::PathBuf};

use bevy::math::Vec3;
use chrono::Local;
use fnv::{FnvHashMap, FnvHashSet};
use ndarray::Array2;
use stl_io::{Normal, Triangle, Vertex};

use crate::Field;

#[derive(Clone, Copy)]
struct Facet(Vec3, Vec3, Vec3);

impl Facet {
    fn normal(&self) -> Normal {
        let v0 = self.1 - self.0;
        let v1 = self.2 - self.0;
        let n = v0.cross(v1).normalize();
        Normal::new([n.x, n.y, n.z])
    }

    fn to_triangle(self) -> Triangle {
        Triangle {
            normal: self.normal(),
            vertices: [
                Vertex::new([self.0.x, self.0.y, self.0.z]),
                Vertex::new([self.1.x, self.1.y, self.1.z]),
                Vertex::new([self.2.x, self.2.y, self.2.z]),
            ],
        }
    }
}

pub fn write_to_stl(field: &Field) -> std::io::Result<PathBuf> {
    let triangles = cells_to_triangles(&field.0.read().cells, 0.025, 0.1);
    let now = Local::now();
    let filename = format!("snowflake-{}.stl", now.format("%Y%m%d%H%M%S"));
    let path = PathBuf::from(&filename);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    stl_io::write_stl(&mut file, triangles.iter())?;
    Ok(path)
}

fn cells_to_triangles(cells: &Array2<f32>, xy_scale: f32, z_scale: f32) -> Vec<Triangle> {
    let n = cells.shape()[0];
    let sqrt3_2 = 3.0f32.sqrt() / 2.0;
    let x_offset = (n as f32 - 1.0) * 1.5 * xy_scale / 2.0;
    let y_offset = (n as f32 - 1.0) * xy_scale * sqrt3_2 / 2.0;
    let mut facets = Vec::new();
    // 辺の集合。始点をキーとして終点の集合を値とする。
    let mut segments: FnvHashMap<(usize, usize), FnvHashSet<(usize, usize)>> =
        FnvHashMap::default();
    let directions0 = [(0, 0), (1, 0), (0, 1)];
    let directions1 = [(1, 0), (1, 1), (0, 1)];

    for i in 0..(n - 1) {
        for j in 0..(n - 1) {
            let i1 = i + 1;
            let j1 = j + 1;
            /*
             (i, j1)  (i1, j1)
                ↓      ↓
                p10 <-- p11
                /   \   /
              p00 --> p01
               ↑     ↑
             (i, j) (i1, j)
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

            // (p00, p01, p10)とその裏
            if cells[[i, j]] > 0.0 && cells[[i1, j]] > 0.0 && cells[[i, j1]] > 0.0 {
                facets.push(Facet(p00, p01, p10));
                // 裏は法線が逆向きなので順番を逆にする
                facets.push(Facet(
                    p10.with_z(-p10.z),
                    p01.with_z(-p01.z),
                    p00.with_z(-p00.z),
                ));
            }
            // (p01, p11, p10)とその裏
            if cells[[i1, j]] > 0.0 && cells[[i1, j1]] > 0.0 && cells[[i, j1]] > 0.0 {
                facets.push(Facet(p01, p11, p10));
                // 裏は法線が逆向きなので順番を逆にする
                facets.push(Facet(
                    p10.with_z(-p10.z),
                    p11.with_z(-p11.z),
                    p01.with_z(-p01.z),
                ));
            }

            // 輪郭抽出のために辺を追加
            if cells[[i, j]] > 0.0 && cells[[i1, j]] > 0.0 && cells[[i, j1]] > 0.0 {
                for k in 0..3 {
                    let start = (i + directions0[k].0, j + directions0[k].1);
                    let end = (
                        i + directions0[(k + 1) % 3].0,
                        j + directions0[(k + 1) % 3].1,
                    );
                    // 逆向きの辺がすでに追加されていたら打ち消して終了、そうでなければ追加
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

    // 輪郭抽出
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

    // 輪郭から側面を生成
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

    // 法線を計算してTriangleに変換
    facets.into_iter().map(Facet::to_triangle).collect()
}
