#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("example does not run in wasm32 mode");
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    todo!("not implemented yet");
    /*
    use hog_detector::{DataSet, HogDetector};
    use object_detector_rust::{dataset::FolderDataSet, prelude::BayesClassifier};

    const ANNOTATIONS: usize = 42;
    const IMAGES_PER_LABEL: usize = 21;

    let mut model: HogDetector<f32, usize, BayesClassifier<f32, usize>, _> = HogDetector::default();

    let data_path = std::fs::canonicalize("res/training/").unwrap();
    let labels_path = std::fs::canonicalize("res/labels.txt").unwrap();
    let label_names = FolderDataSet::load_label_names(labels_path.to_str().unwrap());
    let mut dataset = FolderDataSet::new(data_path.to_str().unwrap(), 32, 32, label_names);

    dataset.load();
    //dataset.augment();

    model.fit_class(&dataset, 5);
    assert_eq!(dataset.samples(), ANNOTATIONS * IMAGES_PER_LABEL);
    dataset.generate_hard_negative_samples(&model, 5, None);
    assert!(dataset.samples() > ANNOTATIONS * IMAGES_PER_LABEL);
    */
}
