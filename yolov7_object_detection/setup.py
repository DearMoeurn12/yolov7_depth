from setuptools import setup

package_name = 'yolov7_object_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        #('share/' + package_name, ['package.xml',
        #                          'launch/object_detection.launch.xml']),
        
        ('lib/' + package_name + '/models/',['models/experimental.py', 'models/common.py',
                                             'models/yolo.py']),
        ('lib/' + package_name + '/utils/',['utils/general.py', 'utils/torch_utils.py',
                                             'utils/plots.py', 'utils/datasets.py',
                                             'utils/google_utils.py', 'utils/activations.py',
                                             'utils/add_nms.py', 'utils/autoanchor.py',
                                             'utils/loss.py', 'utils/metrics.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='muchiro',
    maintainer_email='muchiro@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
         'depth_object_detection = yolov7_object_detection.object_detection_depth:main',
         'pub_bbox_detection = yolov7_object_detection.pub_bboxs_detection:main',
         'fixIOU_detection = yolov7_object_detection.fixIOU_object_detection:main',
         'sub_bboxes= yolov7_object_detection.sub_bboxes:main',
         'torch_detection = yolov7_object_detection.pub_bboxs_detection_with_pytorch:main',
         'sim_yolo = yolov7_object_detection.simulation_detection:main',
        ],
    },
)
