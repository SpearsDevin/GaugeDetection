import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Load the TensorFlow model into memory
model_path = 'C:/Users/devin/Downloads/Gauge Detection/test detection/saved_model/saved_model.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load the label map
label_map_path = 'C:/Users/devin/Downloads/Gauge Detection/test detection/saved_model/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Connect to the RTSP stream
cap = cv2.VideoCapture('rtsp://admin:Midtech31@192.168.2.42:80/0')

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            # Read the next frame from the stream
            ret, frame = cap.read()

            # Run the TensorFlow model on the frame
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame[None, ...]})

            # Draw the detection results on the frame
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                boxes[0],
                classes[0].astype(int),
                scores[0],
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=.5,
                line_thickness=8)

            # Show the frame
            cv2.imshow('RTSP Stream', frame)

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release the stream and close the window
cap.release()
cv2.destroyAllWindows()