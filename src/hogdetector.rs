use crate::detector::visualize_detections;
use crate::Detector;
use image::{DynamicImage, GenericImageView};
use linfa::{Float, Label};
use ndarray::Array2;
use object_detector_rust::prelude::{BBox, Class, Detection, WindowGenerator};
use object_detector_rust::prelude::{Classifier, SlidingWindow};
use object_detector_rust::prelude::{DataSet, Feature, HOGFeature, PersistentDetector};
use object_detector_rust::utils::{evaluate_precision, extract_data};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::error::Error;
use std::io::{Read, Write};
use std::marker::PhantomData;

/// Hog Detector struct
/// ,-----.  ,---------.   ,--------------------.   ,---.                             
/// |Image|  |GrayImage|   |HogFeatureDescriptor|   |SVM|                             
/// |-----|--|---------|---|--------------------|---|---|                             
/// `-----'  `---------'   `--------------------'   `---'                             
///                                                    |                              
///                                                    |                              
///                                             ,-----------.   ,--------.   ,-------.
///                                             |Predictions|   |Detector|   |Objects|
///                                             |-----------|---|--------|---|-------|
///                                             `-----------'   `--------'   `-------'
///
#[derive(Debug)]
pub struct HogDetector<X, Y, C: Classifier<X, Y>, W>
where
    X: Float,
    Y: Label,
    W: WindowGenerator<DynamicImage>,
{
    x: PhantomData<*const X>,
    y: PhantomData<*const Y>,
    /// support vector classifier
    pub classifier: Option<C>,
    /// the feature descriptor
    pub feature_descriptor: Box<dyn Feature>,

    window_generator: W,
}

/// trait of an hog detector
pub trait HogDetectorTrait<X, Y>: Detector + Send + Sync {
    /// fit model for class
    fn fit_class(
        &mut self,
        x: &Vec<DynamicImage>,
        y: &Vec<usize>,
        class: Class,
    ) -> Result<(), String>;
    /// reference to detector trait
    fn detector(&self) -> &dyn Detector;
}

unsafe impl<X, Y, C: Classifier<X, Y>, W> Send for HogDetector<X, Y, C, W>
where
    X: Float,
    Y: Label,
    W: WindowGenerator<DynamicImage>,
{
}
unsafe impl<X, Y, C: Classifier<X, Y>, W> Sync for HogDetector<X, Y, C, W>
where
    X: Float,
    Y: Label,
    W: WindowGenerator<DynamicImage>,
{
}

impl<X, Y, C: Classifier<X, Y>, W> PartialEq for HogDetector<X, Y, C, W>
where
    X: Float,
    Y: Label,
    W: WindowGenerator<DynamicImage>,
{
    fn eq(&self, other: &HogDetector<X, Y, C, W>) -> bool {
        self.classifier.is_none() && other.classifier.is_none()
            || self.classifier.is_some()
                && other.classifier.is_some()
                && self.classifier.eq(&other.classifier)
    }
}

impl<X, Y, C: Classifier<X, Y> + Default> Default for HogDetector<X, Y, C, SlidingWindow>
where
    X: Float,
    Y: Label,
{
    fn default() -> Self {
        HogDetector::<X, Y, C, SlidingWindow> {
            classifier: Some(C::default()),
            feature_descriptor: Box::new(HOGFeature::default()),
            window_generator: SlidingWindow {
                width: 32,
                height: 32,
                step_size: 24,
            },
            x: PhantomData,
            y: PhantomData,
        }
    }
}

impl<X, Y, C: Classifier<X, Y> + object_detector_rust::predictable::Predictable<f32, usize>, W>
    HogDetector<X, Y, C, W>
where
    X: Float,
    Y: Label,
    W: WindowGenerator<DynamicImage>,
{
    /// evaluate class on dataset
    pub fn evaluate(&mut self, dataset: &dyn DataSet, class: Class) -> f32 {
        let (x, y) = dataset.get_data();
        let x: Vec<DynamicImage> = x.into_iter().map(|x| x.thumbnail_exact(32, 32)).collect();
        let x: Vec<Vec<_>> = x
            .iter()
            .map(|image| self.feature_descriptor.extract(image).unwrap())
            .collect();
        let y = y
            .iter()
            .map(|y| if *y == class { class as usize } else { 0 })
            .collect();
        let (x, y) = extract_data(x, y);
        evaluate_precision::<f32, usize>(self.classifier.as_ref().unwrap(), &x, &y)
    }
}

impl<C: Classifier<f32, bool>, W> HogDetector<f32, bool, C, W>
where
    W: WindowGenerator<DynamicImage>,
{
    /// visualizes detections
    pub fn visualize_detections(&self, image: &DynamicImage) -> DynamicImage {
        let detections = self.detect(image);
        visualize_detections(image, &detections)
    }
}

impl<C: Classifier<f32, usize>, W> HogDetector<f32, usize, C, W>
where
    W: WindowGenerator<DynamicImage>,
{
    /// visualizes detections
    pub fn visualize_detections(&self, image: &DynamicImage) -> DynamicImage {
        let detections = self.detect(image);
        visualize_detections(image, &detections)
    }
}

impl<C: Classifier<f32, usize>, W> Detector for HogDetector<f32, usize, C, W>
where
    W: WindowGenerator<DynamicImage>,
{
    fn detect(&self, image: &DynamicImage) -> Vec<Detection> {
        let windows = self.window_generator.windows(image);
        let windows_len = windows.len();
        let hog_features: Vec<Vec<f32>> = windows
            .iter()
            .flat_map(|window| {
                self.feature_descriptor
                    .extract(&DynamicImage::ImageRgba8(window.view.to_image()))
            })
            .collect();

        let features_len = match hog_features.first() {
            Some(features) => features.len(),
            None => 0,
        };
        let hog_features: Vec<f32> = hog_features.into_iter().flatten().collect();
        let hog_features =
            Array2::from_shape_vec((windows_len, features_len), hog_features).unwrap();
        let predictions = self
            .classifier
            .as_ref()
            .unwrap()
            .predict(&hog_features.view())
            .unwrap();
        assert_eq!(windows_len, predictions.len());
        let mut detections = Vec::new();
        for (i, &prediction) in predictions.iter().enumerate() {
            if prediction > 0 {
                let window = windows[i];
                detections.push(Detection::new(
                    BBox::new(
                        window.x as i32,
                        window.y as i32,
                        window.view.width(),
                        window.view.height(),
                    ),
                    1 as Class,
                    1.0,
                ));
            }
        }
        detections
    }
}

impl<C: Classifier<f32, bool>, W> Detector for HogDetector<f32, bool, C, W>
where
    W: WindowGenerator<DynamicImage>,
{
    fn detect(&self, image: &DynamicImage) -> Vec<Detection> {
        let windows = self.window_generator.windows(image);
        let windows_len = windows.len();
        let hog_features: Vec<Vec<f32>> = windows
            .iter()
            .flat_map(|window| {
                self.feature_descriptor
                    .extract(&DynamicImage::ImageRgba8(window.view.to_image()))
            })
            .collect();

        let features_len = match hog_features.first() {
            Some(features) => features.len(),
            None => 0,
        };
        let hog_features: Vec<f32> = hog_features.into_iter().flatten().collect();
        let hog_features =
            Array2::from_shape_vec((windows_len, features_len), hog_features).unwrap();
        let predictions = self
            .classifier
            .as_ref()
            .unwrap()
            .predict(&hog_features.view())
            .unwrap();
        assert_eq!(windows_len, predictions.len());
        let mut detections = Vec::new();
        for (i, &prediction) in predictions.iter().enumerate() {
            if prediction {
                let window = windows[i];
                detections.push(Detection::new(
                    BBox::new(
                        window.x as i32,
                        window.y as i32,
                        window.view.width(),
                        window.view.height(),
                    ),
                    1 as Class,
                    1.0,
                ));
            }
        }
        detections
    }
}

impl<C: Classifier<f32, bool> + Serialize + DeserializeOwned, WG> PersistentDetector
    for HogDetector<f32, bool, C, WG>
where
    WG: WindowGenerator<DynamicImage>,
{
    fn save<W: Write>(&self, mut writer: W) -> Result<(), Box<dyn Error>> {
        // Serialize the SVMClassifier using the `bincode` crate
        let svm_classifier_bytes = bincode::serialize(&self.classifier.as_ref().unwrap())?;

        // Write the serialized bytes to the writer
        writer.write_all(&svm_classifier_bytes)?;

        Ok(())
    }

    fn load<R: Read>(&mut self, mut reader: R) -> Result<(), Box<dyn Error>> {
        // Read the serialized bytes from the reader
        let mut classifier_bytes = Vec::new();
        reader.read_to_end(&mut classifier_bytes)?;

        // Deserialize the bytes using the `bincode` crate
        let classifier = bincode::deserialize(&classifier_bytes)?;
        self.classifier = Some(classifier);

        Ok(())
    }
}

impl<C: Classifier<f32, usize> + Serialize + DeserializeOwned, WG> PersistentDetector
    for HogDetector<f32, usize, C, WG>
where
    WG: WindowGenerator<DynamicImage>,
{
    fn save<W: Write>(&self, mut writer: W) -> Result<(), Box<dyn Error>> {
        // Serialize the SVMClassifier using the `bincode` crate
        let svm_classifier_bytes = bincode::serialize(&self.classifier.as_ref().unwrap())?;

        // Write the serialized bytes to the writer
        writer.write_all(&svm_classifier_bytes)?;

        Ok(())
    }

    fn load<R: Read>(&mut self, mut reader: R) -> Result<(), Box<dyn Error>> {
        // Read the serialized bytes from the reader
        let mut classifier_bytes = Vec::new();
        reader.read_to_end(&mut classifier_bytes)?;

        // Deserialize the bytes using the `bincode` crate
        let classifier = bincode::deserialize(&classifier_bytes)?;
        self.classifier = Some(classifier);

        Ok(())
    }
}

impl<C: Classifier<f32, usize>, WG> HogDetectorTrait<f32, usize> for HogDetector<f32, usize, C, WG>
where
    WG: WindowGenerator<DynamicImage>,
{
    /// fit model for class
    fn fit_class(
        &mut self,
        x: &Vec<DynamicImage>,
        y: &Vec<usize>,
        class: Class,
    ) -> Result<(), String> {
        let x: Vec<Vec<f32>> = x
            .iter()
            .map(|image| self.feature_descriptor.extract(image).unwrap())
            .collect();
        let y = y
            .iter()
            .map(|y| {
                if *y == class as usize {
                    class as usize
                } else {
                    0
                }
            })
            .collect();
        let (x, y) = extract_data(x, y);
        self.classifier.as_mut().unwrap().fit(&x.view(), &y.view())
    }

    fn detector(&self) -> &dyn Detector {
        self
    }
}

impl<C: Classifier<f32, bool>, WG> HogDetectorTrait<f32, bool> for HogDetector<f32, bool, C, WG>
where
    WG: WindowGenerator<DynamicImage>,
{
    /// fit model for class
    fn fit_class(
        &mut self,
        x: &Vec<DynamicImage>,
        y: &Vec<usize>,
        class: Class,
    ) -> Result<(), String> {
        let x: Vec<Vec<f32>> = x
            .iter()
            .map(|image| self.feature_descriptor.extract(image).unwrap())
            .collect();
        let y = y
            .iter()
            .map(|y| if *y == class as usize { true } else { false })
            .collect();
        let (x, y) = extract_data(x, y);
        self.classifier.as_mut().unwrap().fit(&x.view(), &y.view())
    }

    fn detector(&self) -> &dyn Detector {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::BayesClassifier;
    use image::Rgb;
    use object_detector_rust::{
        prelude::{MemoryDataSet, SVMClassifier},
        tests::test_image,
    };

    #[test]
    fn test_default() {
        let model = HogDetector::<f32, bool, SVMClassifier<f32, bool>, SlidingWindow>::default();
        assert!(model.classifier.is_some());
    }

    #[test]
    fn test_part_eq() {
        let model1 = HogDetector::<f32, bool, SVMClassifier<f32, bool>, SlidingWindow>::default();
        let model2 = HogDetector::<f32, bool, SVMClassifier<f32, bool>, SlidingWindow>::default();
        assert!(model1.classifier.is_some());
        assert!(model1.eq(&model2));
        assert!(model1.eq(&model1));
    }

    #[test]
    fn test_evaluate() {
        let mut model: HogDetector<f32, usize, BayesClassifier<f32, usize>, _> =
            HogDetector::default();

        let mut dataset = MemoryDataSet::new_test();
        dataset.load().unwrap();
        let (x, y) = dataset.get_data();
        let x = x.into_iter().map(|x| x.thumbnail_exact(32, 32)).collect();
        let y = y.into_iter().map(|y| y as usize).collect::<Vec<_>>();

        model.fit_class(&x, &y, 1).unwrap();
        assert!(model.classifier.is_some());
        assert!(model.evaluate(&dataset, 1) > 0.0);
    }

    #[test]
    fn test_bool_detector() {
        let img = test_image();
        let mut dataset = MemoryDataSet::new_test();
        dataset.load().unwrap();
        let (x, y) = dataset.get_data();
        let x = x.into_iter().map(|x| x.thumbnail_exact(32, 32)).collect();
        let y = y.into_iter().map(|y| y as usize).collect::<Vec<_>>();

        let mut detector: HogDetector<f32, bool, SVMClassifier<f32, bool>, _> =
            HogDetector::default();
        detector.fit_class(&x, &y, 1).unwrap();
        let detections = detector.detect(&img);
        assert!(detections.is_empty());
        let visualization = detector.visualize_detections(&img).to_rgb8();
        assert_eq!(&Rgb([0, 0, 0]), visualization.get_pixel(55, 0));
        assert_eq!(&Rgb([255, 0, 0]), visualization.get_pixel(75, 0));
    }

    #[test]
    fn test_save_load_usize() {
        let mut model: HogDetector<f32, usize, BayesClassifier<_, _>, _> = HogDetector::default();
        let mut dataset = MemoryDataSet::new_test();
        dataset.load().unwrap();
        let (x, y) = dataset.get_data();
        let x = x.into_iter().map(|x| x.thumbnail_exact(32, 32)).collect();
        let y = y.into_iter().map(|y| y as usize).collect::<Vec<_>>();

        model.fit_class(&x, &y, 1).unwrap();
        let mut serialized = Vec::new();
        model.save(&mut serialized).unwrap();
        let mut model2: HogDetector<f32, usize, BayesClassifier<_, _>, _> = HogDetector::default();
        model2.load(&mut &serialized[..]).unwrap();
        assert_eq!(model, model2);
    }

    #[test]
    fn test_save_load_bool() {
        let mut model: HogDetector<f32, bool, SVMClassifier<_, _>, _> = HogDetector::default();
        let mut dataset = MemoryDataSet::new_test();
        dataset.load().unwrap();
        let (x, y) = dataset.get_data();
        let x = x.into_iter().map(|x| x.thumbnail_exact(32, 32)).collect();
        let y = y.into_iter().map(|y| y as usize).collect::<Vec<_>>();

        model.fit_class(&x, &y, 1).unwrap();
        let mut serialized = Vec::new();
        model.save(&mut serialized).unwrap();
        let mut model2: HogDetector<f32, bool, SVMClassifier<_, _>, _> = HogDetector::default();
        model2.load(&mut &serialized[..]).unwrap();
        assert_eq!(model, model2);
    }
}
