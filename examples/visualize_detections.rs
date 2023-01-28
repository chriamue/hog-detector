#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("example does not run in wasm32 mode");
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use hog_detector::{DataSet, HogDetector};
    use object_detector_rust::{
        dataset::FolderDataSet, prelude::BayesClassifier, prelude::Predictable,
    };

    let mut model: HogDetector<f32, bool, BayesClassifier<f32, bool>, _> = HogDetector::default();

    let data_path = std::fs::canonicalize("res/training/").unwrap();
    let labels_path = std::fs::canonicalize("res/labels.txt").unwrap();
    let label_names = FolderDataSet::load_label_names(labels_path.to_str().unwrap());
    let mut dataset = FolderDataSet::new(data_path.to_str().unwrap(), 32, 32, label_names);
    dataset.load().unwrap();
    //dataset.generate_random_annotations(10);
    let (x, y) = dataset.get_data();
    model.fit_class(&x, &y, 1).unwrap();
    assert!(model.classifier.is_some());
    let webcam01 = image::open("res/training/webcam01.jpg").unwrap();
    model
        .visualize_detections(&webcam01)
        .save("out/test_visualize_detections_1.png")
        .unwrap();
    let (x, y) = dataset.get_data();
    model.fit_class(&x, &y, 2).unwrap();
    let webcam06 = image::open("res/training/webcam06.jpg").unwrap();
    model
        .visualize_detections(&webcam06)
        .save("out/test_visualize_detections_2.png")
        .unwrap();
    let (x, y) = dataset.get_data();
    model.fit_class(&x, &y, 5).unwrap();
    //dataset.generate_hard_negative_samples(&model, 5, None);
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
