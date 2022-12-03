#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("example does not run in wasm32 mode");
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use hog_detector::classifier::SVMClassifier;
    use hog_detector::data_augmentation::DataAugmentation;
    use hog_detector::dataset::{DataGenerator, DataSet, FolderDataSet};
    use hog_detector::HogDetector;
    use hog_detector::Trainable;

    const ANNOTATIONS: usize = 42;
    const IMAGES_PER_LABEL: usize = 21;

    let mut model: HogDetector<SVMClassifier> = HogDetector::default();

    let mut dataset = FolderDataSet::new(
        "res/training/".to_string(),
        "res/labels.txt".to_string(),
        32,
    );
    dataset.load();
    dataset.augment();

    model.train_class(&dataset, 5);
    assert_eq!(dataset.samples(), ANNOTATIONS * IMAGES_PER_LABEL);
    dataset.generate_hard_negative_samples(&model, 5, None);
    assert!(dataset.samples() > ANNOTATIONS * IMAGES_PER_LABEL);
}
