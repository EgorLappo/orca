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

    pub fn write_row(
        &mut self,
        id: String,
        orca_marker: f64,
        orca_full: Option<f64>,
        orca_sim: Option<f64>,
    ) -> Result<()> {
        self.writer
            .serialize(OutRow {
                id,
                orca_marker,
                orca_full,
                orca_sim,
            })
            .wrap_err("failed to write output row")?;
        self.writer.flush().wrap_err("failed to flush writer")?;
        Ok(())
    }

    pub fn write_rows(
        &mut self,
        mut rows: impl Iterator<Item = (String, f64, Option<f64>, Option<f64>)>,
    ) -> Result<()> {
        rows.try_for_each(|(id, orca_marker, orca_full, orca_sim)| {
            self.writer
                .serialize(OutRow {
                    id,
                    orca_marker,
                    orca_full,
                    orca_sim,
                })
                .wrap_err("failed to write output row")
        })
    }
}

// little struct to write csv easier
#[derive(Debug, serde::Serialize)]
struct OutRow {
    id: String,
    orca_marker: f64,
    orca_full: Option<f64>,
    orca_sim: Option<f64>,
}
