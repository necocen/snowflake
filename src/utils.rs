use fnv::FnvHashMap;
use ndarray::Array2;

pub fn extract_contours(grid: &Array2<bool>, scale: f32) -> Vec<Vec<(f32, f32)>> {
    let mut segments: FnvHashMap<(i32, i32), (i32, i32)> = FnvHashMap::default();
    let directions = [(1, 1), (0, 2), (-1, 1), (-1, -1), (0, -2), (1, -1)];
    grid.indexed_iter().for_each(|((i, j), v)| {
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
                        scale * x as f32 / 2.0,
                        scale * y as f32 / 2.0 / 3.0f32.sqrt(),
                    )
                })
                .collect()
        })
        .collect()
}
