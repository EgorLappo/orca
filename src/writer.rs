use color_eyre::eyre::{Result, WrapErr};
use std::fs::File;
use std::path::Path;

pub struct OutputWriter {
    writer: csv::Writer<File>,
}

impl OutputWriter {
    pub fn new(output: &Path) -> Result<Self> {
        let writer = csv::Writer::from_path(output)?;

        Ok(OutputWriter { writer })
    }

    pub fn write_row(&mut self, id: String, orca: f64) -> Result<()> {
        self.writer
            .serialize(OutRow { id, orca })
            .wrap_err("failed to write output row")
    }
}

#[derive(Debug, serde::Serialize)]
struct OutRow {
    id: String,
    orca: f64,
}
