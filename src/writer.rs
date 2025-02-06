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

    pub fn write_row(&mut self, row: OutRow) -> Result<()> {
        self.writer
            .serialize(row)
            .wrap_err("failed to write output row")?;
        self.writer.flush().wrap_err("failed to flush writer")?;
        Ok(())
    }

    pub fn write_rows(&mut self, mut rows: impl Iterator<Item = OutRow>) -> Result<()> {
        rows.try_for_each(|row| {
            self.writer
                .serialize(row)
                .wrap_err("failed to write output row")
        })
    }
}

// little struct to write csv easier
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct OutRow {
    pub id: String,
    pub orca_marker: f64,
    pub orca_full: Option<f64>,
    pub orca_sim: Option<f64>,
}
