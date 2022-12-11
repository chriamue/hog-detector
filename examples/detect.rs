#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("example does not run in wasm32 mode");
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use hog_detector::classifier::BayesClassifier;
    use hog_detector::dataset::{DataSet, FolderDataSet};
    use hog_detector::Detector;
    use hog_detector::HogDetector;
    use hog_detector::Trainable;
    let mut model: HogDetector<BayesClassifier> = HogDetector::default();

    let mut dataset = FolderDataSet::new(
        "res/training/".to_string(),
        "res/labels.txt".to_string(),
        32,
    );
    dataset.load();

    model.train_class(&dataset, 5);
    assert!(model.classifier.is_some());
    let webcam01 = image::open("res/training/webcam01.jpg").unwrap();
    let detections = model.detect_objects(&webcam01);
    println!("{:?}", detections);
    assert!(!detections.is_empty());
}
