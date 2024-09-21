use std::{
    io::{Cursor, Write as _},
    path::PathBuf,
};

use chrono::{DateTime, Local};
use resvg::{
    tiny_skia::Pixmap,
    usvg::{Options, Transform, Tree},
};

use crate::{svg::cells_to_document, Field};

pub fn write_to_png(field: &Field, now: DateTime<Local>) -> anyhow::Result<PathBuf> {
    let size = 2000;
    let document = cells_to_document(&field.0.read().cells, size as f32);
    let mut buffer = Vec::new();
    let mut writer = Cursor::new(&mut buffer);
    svg::write(&mut writer, &document)?;
    writer.flush()?;
    let tree = Tree::from_data(&buffer, &Options::default())?;
    let mut pixmap = Pixmap::new(size, size).unwrap();
    resvg::render(&tree, Transform::default(), &mut pixmap.as_mut());
    let filename = format!("snowflake-{}.png", now.format("%Y%m%d%H%M%S"));
    let path = PathBuf::from(&filename);
    pixmap.save_png(&path)?;
    Ok(path)
}
