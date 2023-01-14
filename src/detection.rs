// source: https://github.com/12101111/yolo-rs/blob/master/src/yolo.rs
use float_ord::FloatOrd;
use object_detector_rust::bbox::BBox;

/// representation of a detection
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// bounding box of the detection
    pub bbox: BBox,
    /// class of the detection
    pub class: usize,
    /// confidence of the detection
    pub confidence: f32,
}

const NMS_THRESH: f32 = 0.45;

impl Detection {
    /// merge other detection into this detection
    pub fn merge(&mut self, other: &Detection) {
        if self.class == other.class && self.bbox.overlap(&other.bbox) > 0.0 {
            self.bbox.width = (0.5 * self.bbox.width as f32
                + 0.5 * other.bbox.width as f32
                + (self.bbox.x - other.bbox.x).abs() as f32) as u32;
            self.bbox.height = (0.5 * self.bbox.height as f32
                + 0.5 * other.bbox.height as f32
                + (self.bbox.y - other.bbox.y).abs() as f32) as u32;
            self.bbox.x = ((self.bbox.x + other.bbox.x) as f32 / 2.0) as i32;
            self.bbox.y = ((self.bbox.y + other.bbox.y) as f32 / 2.0) as i32;
        }
    }
}

/// merge all possible detections from the list of detections
pub fn merge(dets: Vec<Detection>) -> Vec<Detection> {
    let mut merged = Vec::new();
    dets.clone().into_iter().for_each(|d| {
        let mut merged_d = d;
        dets.iter().copied().for_each(|d2| merged_d.merge(&d2));
        merged.push(merged_d);
    });
    merged
}

/// run non max suppression sort algorithm on detections
pub fn nms_sort(mut dets: Vec<Detection>) -> Vec<Detection> {
    let mut ans = Vec::new();
    while !dets.is_empty() {
        dets.sort_by_key(|d| FloatOrd(d.confidence));
        ans.push(dets.pop().unwrap());
        dets = dets
            .into_iter()
            .filter(|d| d.bbox.overlap(&ans.last().unwrap().bbox) < NMS_THRESH)
            .collect();
    }
    ans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nms_sort_test() {
        let det1 = Detection {
            bbox: BBox {
                x: 0.5,
                y: 0.5,
                w: 1.0,
                h: 1.0,
            },
            class: 0,
            confidence: 0.1,
        };
        let det2 = Detection {
            bbox: BBox {
                x: 0.6,
                y: 0.6,
                w: 1.0,
                h: 1.0,
            },
            class: 0,
            confidence: 0.1,
        };
        let det3 = Detection {
            bbox: BBox {
                x: 1.5,
                y: 1.5,
                w: 1.0,
                h: 1.0,
            },
            class: 1,
            confidence: 0.1,
        };
        let detections = vec![det1, det2, det3];
        let sorted = nms_sort(detections);
        assert_eq!(sorted.len(), 2);
        println!("{:?}", sorted);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn merge_test() {
        let mut det1 = Detection {
            bbox: BBox {
                x: 0.4,
                y: 0.4,
                w: 1.0,
                h: 1.0,
            },
            class: 0,
            confidence: 0.1,
        };
        let det2 = Detection {
            bbox: BBox {
                x: 0.6,
                y: 0.6,
                w: 1.0,
                h: 1.0,
            },
            class: 0,
            confidence: 0.1,
        };
        det1.merge(&det2);
        assert_eq!(det1.bbox.x, 0.5);
        assert_eq!(det1.bbox.y, 0.5);
        assert_eq!(det1.bbox.w, 1.2);
        assert_eq!(det1.bbox.h, 1.2);
        let det3 = Detection {
            bbox: BBox {
                x: 4.6,
                y: 4.6,
                w: 1.0,
                h: 1.0,
            },
            class: 0,
            confidence: 0.1,
        };
        det1.merge(&det3);
        assert_eq!(det1.bbox.x, 0.5);
        assert_eq!(det1.bbox.y, 0.5);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn merge_all_test() {
        let det1 = Detection {
            bbox: BBox {
                x: 0.4,
                y: 0.4,
                w: 1.0,
                h: 1.0,
            },
            class: 0,
            confidence: 0.1,
        };
        let det2 = Detection {
            bbox: BBox {
                x: 0.6,
                y: 0.6,
                w: 1.0,
                h: 1.0,
            },
            class: 0,
            confidence: 0.1,
        };
        let dets = vec![det1, det2];
        let merged = merge(dets);

        assert_eq!(merged[0].bbox.y, 0.5);
    }
}
