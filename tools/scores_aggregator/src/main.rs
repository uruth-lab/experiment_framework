// TODO 4: Need to make the easier to set and possibly not relative to current folder
//      - maybe save in default config location and only ask if not found and then have cli arg to override
const RESULTS_FOLDER: &str = "../../results";

const FILTER_SEARCH_KEY: &str = "_performances.yaml";
const OUTPUT_FN: &str = "collected_scores.csv";

use anyhow::{bail, Context};
use serde_yaml::Value;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut file_list = std::fs::read_dir(RESULTS_FOLDER)
        .with_context(|| format!("failed to get directory listing for {:?}", RESULTS_FOLDER))?
        .map(|x| Ok(x.context("failed to open read_dir path")?.path()))
        .filter(|x| {
            x.as_ref().is_ok_and(|x| {
                x.is_file()
                    && x.file_name()
                        .expect("filename format is expected to be utf8")
                        .to_string_lossy()
                        .contains(FILTER_SEARCH_KEY)
            })
        })
        .collect::<anyhow::Result<Vec<PathBuf>>>()?;

    file_list.sort_unstable();

    let output_fn = PathBuf::from(RESULTS_FOLDER).join(OUTPUT_FN);
    let output_file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&output_fn)?;
    let mut csv_wtr = csv::Writer::from_writer(output_file);
    println!("Writing CSV output to {output_fn:?}");

    for file in file_list.iter().rev() {
        let current_filename = file
            .file_name()
            .expect("valid filename expected")
            .to_string_lossy();
        println!("File: {current_filename:?}");
        let f =
            std::fs::File::open(file).with_context(|| format!("failed to open file {file:?}"))?;
        let val: Value = serde_yaml::from_reader(f)?;
        if let Value::Mapping(map) = val {
            for (experiment, performances) in map {
                let experiment = experiment.as_str().expect("all keys should be strings");
                if let Some(Value::Sequence(score_batches)) = performances.get("Scores") {
                    println!("{experiment}");
                    for (trail_number, score_batch) in (1..).zip(score_batches.iter()) {
                        if let Value::Sequence(scores_values) = score_batch {
                            let scores: Vec<f64> = scores_values
                                .iter()
                                .map(|x| match x.as_f64() {
                                    Some(v) => Ok(v),
                                    None => Err(anyhow::anyhow!("expected a number but got {x:?}")),
                                })
                                .collect::<anyhow::Result<Vec<f64>>>()?;
                            println!("{trail_number}-{scores:?}");

                            // Now that we know score values is all numbers insert the experiment name before adding to CSV
                            let mut output_record = Vec::with_capacity(scores_values.len() + 3);
                            output_record.push(Value::String(current_filename.to_string()));
                            output_record
                                .push(Value::Number(serde_yaml::Number::from(trail_number)));
                            output_record.push(Value::String(experiment.to_string()));
                            scores_values
                                .iter()
                                .cloned()
                                .for_each(|x| output_record.push(x));
                            csv_wtr.serialize(output_record)?;
                            // TODO 4: Decide how to deal with different number of points in datasets. (Ignore all not same as first?)
                        } else {
                            bail!("Non Sequence Batch Found")
                        }
                    }
                    println!(); // Space between experiments
                } else {
                    // No scores, skip
                }
            }
        } else {
            println!("Content was not a map");
        }

        println!("{}\n", "-".repeat(40));
    }

    csv_wtr.flush()?;
    println!("CSV Output to {output_fn:?} flushed");

    Ok(())
}
