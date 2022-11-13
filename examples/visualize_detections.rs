#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("example does not run in wasm32 mode");
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use hog_detector::dataset::{DataSet, FolderDataSet};
    use hog_detector::Detector;
    use hog_detector::HogDetector;
    use hog_detector::Trainable;

    let mut model = HogDetector::default();

    let mut dataset = FolderDataSet::new(
        "res/training/".to_string(),
        "res/labels.txt".to_string(),
        32,
    );
    dataset.load(false);
    dataset.generate_random_annotations(10);

    model.train_class(&dataset, 1);
    assert!(model.svc.is_some());
    let webcam01 = image::open("res/training/webcam01.jpg").unwrap();
    model
        .visualize_detections(&webcam01)
        .save("out/test_visualize_detections_1.png")
        .unwrap();

    model.train_class(&dataset, 2);
    let webcam06 = image::open("res/training/webcam06.jpg").unwrap();
    model
        .visualize_detections(&webcam06)
        .save("out/test_visualize_detections_2.png")
        .unwrap();
    model.train_class(&dataset, 5);
    dataset.generate_hard_negative_samples(&model, 5, None);
    let webcam10 = image::open("res/training/webcam01.jpg").unwrap();
    model
        .visualize_detections(&webcam10)
        .save("out/test_visualize_detections_5_01.png")
        .unwrap();
    let webcam10 = image::open("res/training/webcam06.jpg").unwrap();
    model
        .visualize_detections(&webcam10)
        .save("out/test_visualize_detections_5_06.png")
        .unwrap();
    let webcam10 = image::open("res/training/webcam10.jpg").unwrap();
    model
        .visualize_detections(&webcam10)
        .save("out/test_visualize_detections_5_10.png")
        .unwrap();
}
