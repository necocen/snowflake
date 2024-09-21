use std::{fs::OpenOptions, path::PathBuf};

use chrono::{DateTime, Local};
use fnv::FnvHashMap;
use ndarray::Array2;
use svg::node::element::{path::Data, Group, Path};

use crate::Field;

pub fn write_to_svg(field: &Field, now: DateTime<Local>) -> std::io::Result<PathBuf> {
    let document = cells_to_document(&field.0.read().cells, 1000.0);
    let filename = format!("snowflake-{}.svg", now.format("%Y%m%d%H%M%S"));
    let path = PathBuf::from(&filename);
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    svg::write(file, &document)?;
    Ok(path)
}

pub fn cells_to_document(cells: &Array2<f32>, size: f32) -> svg::Document {
    let n = cells.shape()[0];
    let mut group = Group::new();
    let scale = size * 2.0 / 3.0f32.sqrt() / n as f32;
    let max = cells.fold(0.0f32, |a, &b| a.max(b));
    let min = cells.fold(max, |a, &b| if b > 0.0 { a.min(b) } else { a });
    let quantized =
        ((cells - min) / (max - min)).mapv(|v| if v < 0.0 { 0 } else { 255 - (v * 254.0) as u8 });
    for i in 1..=255 {
        let alpha = i as f32 / 255.0;
        let contours = extract_contours(&quantized.mapv(|v| v == i), scale);
        let mut data = Data::new();
        for contour in contours {
            let mut iter = contour.iter();
            if let Some((x, y)) = iter.next() {
                data = data.move_to((*x, *y));
            }
            for (x, y) in iter {
                data = data.line_to((*x, *y));
            }
            data = data.close();
        }
        let path = Path::new()
            .set("d", data)
            .set("fill", "white")
            .set("fill-opacity", alpha);
        group = group.add(path);
    }

    group = group.set(
        "transform",
        format!("rotate(30, {}, {})", size / 2.0, size / 2.0),
    );

    let mut document = svg::Document::new()
        .set("viewBox", (0, 0, size, size))
        .set("width", size)
        .set("height", size);
    document = document.add(group);
    document
}

fn extract_contours(cells: &Array2<bool>, scale: f32) -> Vec<Vec<(f32, f32)>> {
    let n = cells.shape()[0];
    let mut segments: FnvHashMap<(i32, i32), (i32, i32)> = FnvHashMap::default();
    let directions = [(1, 1), (0, 2), (-1, 1), (-1, -1), (0, -2), (1, -1)];
    cells.indexed_iter().for_each(|((i, j), v)| {
        if *v {
            for (k, &(di, dj)) in directions.iter().enumerate() {
                let start = (2 * i as i32 + j as i32 + di, 3 * j as i32 + dj);
                let end = (
                    2 * i as i32 + j as i32 + directions[(k + 1) % 6].0,
                    3 * j as i32 + directions[(k + 1) % 6].1,
                );

                if let Some(&reverse_start) = segments.get(&end) {
                    if reverse_start == start {
                        segments.remove(&end);
                    } else {
                        segments.insert(start, end);
                    }
                } else {
                    segments.insert(start, end);
                }
            }
        }
    });

    let mut contours = Vec::new();
    while !segments.is_empty() {
        let start = *segments.keys().next().unwrap();
        let mut contour = vec![start];
        let mut current = start;
        while let Some(&next) = segments.get(&current) {
            contour.push(next);
            segments.remove(&current);
            current = next;
        }
        contours.push(contour);
    }

    contours
        .into_iter()
        .map(|contour| {
            contour
                .into_iter()
                .map(|(x, y)| {
                    (
                        scale * (x as f32 / 2.0 - ((3.0 - 3.0f32.sqrt()) / 4.0) * n as f32),
                        scale * (n as f32 - y as f32 / 3.0) * 3.0f32.sqrt() / 2.0,
                    )
                })
                .collect()
        })
        .collect()
}
