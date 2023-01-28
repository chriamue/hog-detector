#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("dataset example does not run in wasm32 mode");
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use hog_detector::{DataSet, HogDetector};
    use object_detector_rust::{classifier::BayesClassifier, dataset::FolderDataSet};
    let mut model: HogDetector<f32, bool, BayesClassifier<f32, bool>, _> = HogDetector::default();

    let data_path = std::fs::canonicalize("res/training/").unwrap();
    let labels_path = std::fs::canonicalize("res/labels.txt").unwrap();
    let label_names = FolderDataSet::load_label_names(labels_path.to_str().unwrap());
    let mut dataset = FolderDataSet::new(data_path.to_str().unwrap(), 32, 32, label_names);

    dataset.load().unwrap();
    // dataset.augment();
    let (x, y) = dataset.get_data();
    model.fit_class(&x, &y, 5).unwrap();
    // println!("evaluated: {:?} %", model.evaluate(&dataset, 5) * 100.0);

    //dataset.generate_hard_negative_samples(&model, 5, None);
    model.fit_class(&x, &y, 5).unwrap();
    //println!(
    //    "evaluated after training with hard negative samples: {:?} %",
    //    model.evaluate(&dataset, 5) * 100.0
    //);
}
