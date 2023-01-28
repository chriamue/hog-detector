#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("example does not run in wasm32 mode");
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use hog_detector::{DataSet, HogDetector};
    use object_detector_rust::{dataset::FolderDataSet, prelude::BayesClassifier};
    let mut model: HogDetector<f32, usize, BayesClassifier<f32, usize>, _> = HogDetector::default();

    let data_path = std::fs::canonicalize("res/training/").unwrap();
    let labels_path = std::fs::canonicalize("res/labels.txt").unwrap();
    let label_names = FolderDataSet::load_label_names(labels_path.to_str().unwrap());
    let mut dataset = FolderDataSet::new(data_path.to_str().unwrap(), 32, 32, label_names);
    dataset.load().unwrap();
    let (x, y) = dataset.get_data();
    //dataset.augment();
    let y = y.into_iter().map(|y| y as usize).collect();
    model.fit_class(&x, &y, 5).unwrap();
    assert!(model.classifier.is_some());
    println!("{:?}", model.evaluate(&dataset, 5));
    assert!(model.evaluate(&dataset, 5) > 0.0);
}
