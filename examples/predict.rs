#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("example does not run in wasm32 mode");
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use hog_detector::dataset::{DataSet, FolderDataSet};
    use hog_detector::utils::window_crop;
    use hog_detector::HogDetector;
    use hog_detector::Predictable;
    use hog_detector::Trainable;
    let mut model = HogDetector::default();

    let mut dataset = FolderDataSet::new(
        "res/training/".to_string(),
        "res/labels.txt".to_string(),
        32,
    );
    dataset.load();
    dataset.generate_random_annotations(5);

    model.train_class(&dataset, 5);
    assert!(model.svc.is_some());
    let loco03 = image::open("res/loco03.jpg").unwrap().to_rgb8();
    let loco03 = window_crop(&loco03, 32, 32, (60, 35));

    let predicted = model.predict(&loco03);
    assert_eq!(predicted, 5);
}
