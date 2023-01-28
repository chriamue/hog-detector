#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("example does not run in wasm32 mode");
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use hog_detector::prelude::Predictable;
    use hog_detector::{prelude::BBox, DataSet, HogDetector};
    use ndarray::{Array1, Array2};
    use object_detector_rust::{
        dataset::FolderDataSet, prelude::BayesClassifier, utils::crop_bbox,
    };

    let mut model: HogDetector<f32, usize, BayesClassifier<f32, usize>, _> = HogDetector::default();

    let data_path = std::fs::canonicalize("res/training/").unwrap();
    let labels_path = std::fs::canonicalize("res/labels.txt").unwrap();
    let label_names = FolderDataSet::load_label_names(labels_path.to_str().unwrap());
    let mut dataset = FolderDataSet::new(data_path.to_str().unwrap(), 32, 32, label_names);

    dataset.load().unwrap();
    //dataset.generate_random_annotations(5);

    let (x, y) = dataset.get_data();
    let y = y.into_iter().map(|y| y as usize).collect::<Vec<_>>();
    model.fit_class(&x, &y, 5).unwrap();
    assert!(model.classifier.is_some());
    let loco03 = image::open("res/loco03.jpg").unwrap();
    let loco03 = crop_bbox(&loco03, &BBox::new(60, 15, 32, 32));
    let hog_features = model.feature_descriptor.extract(&loco03).unwrap();
    let x = Array2::from_shape_vec((1, hog_features.len()), hog_features).unwrap();

    let predicted = model
        .classifier
        .as_ref()
        .unwrap()
        .predict(&x.view())
        .unwrap();
    assert_eq!(predicted, Array1::from_vec(vec![5]));
}
