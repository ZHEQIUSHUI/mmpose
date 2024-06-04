dataset_info = dict(
    dataset_name='custom',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='0', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='1',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='3'),
        2:
        dict(
            name='2',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='4'),
        3:
        dict(
            name='3',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='1'),
        4:
        dict(
            name='4',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='2'),
        5:
        dict(
            name='5',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='')
    },
    skeleton_info={
        0:
        dict(link=('0', '5'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('1', '2'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('3', '4'), id=2, color=[255, 128, 0])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1.
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.025, 0.025, 0.025
    ])
