#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("example does not run in wasm32 mode");
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use hog_detector::data_augmentation::DataAugmentation;
    use hog_detector::dataset::{DataSet, FolderDataSet};
    use hog_detector::HogDetector;
    use hog_detector::Trainable;
    let mut model = HogDetector::default();

    let mut dataset = FolderDataSet::new(
        "res/training/".to_string(),
        "res/labels.txt".to_string(),
        32,
    );
    dataset.load();
    dataset.augment();

    model.train_class(&dataset, 5);
    assert!(model.svc.is_some());
    println!("{:?}", model.evaluate(&dataset, 5));
    assert!(model.evaluate(&dataset, 5) > 0.0);
}
