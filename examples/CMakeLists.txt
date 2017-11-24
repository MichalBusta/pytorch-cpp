cmake_minimum_required(VERSION 3.0)
project(example)

ADD_EXECUTABLE(read_allocated_gpu_memory read_allocated_gpu_memory.cpp)
TARGET_LINK_LIBRARIES(read_allocated_gpu_memory ${ATEN_LIBS} ${CUDA_LIBRARIES})

ADD_EXECUTABLE(pytorch_results_deviation pytorch_results_deviation.cpp ${SRC})
TARGET_LINK_LIBRARIES(pytorch_results_deviation pytorch)

# ADD_EXECUTABLE(opencv_realtime_webcam_human_segmentation opencv_realtime_webcam_human_segmentation.cpp)
# TARGET_LINK_LIBRARIES(opencv_realtime_webcam_human_segmentation ${ATen_LIBS} ${OpenCV_LIBS} ${CUDA_LIBRARIES})

# ADD_EXECUTABLE(opencv_realtime_webcam_imagenet_classification opencv_realtime_webcam_imagenet_classification.cpp)
# TARGET_LINK_LIBRARIES(opencv_realtime_webcam_imagenet_classification ${ATen_LIBS} ${OpenCV_LIBS} ${CUDA_LIBRARIES})