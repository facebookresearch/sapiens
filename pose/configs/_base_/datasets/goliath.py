# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

coco_wholebody_info = dict(
    dataset_name='coco_wholebody',
    paper_info=dict(
        author='Jin, Sheng and Xu, Lumin and Xu, Jin and '
        'Wang, Can and Liu, Wentao and '
        'Qian, Chen and Ouyang, Wanli and Luo, Ping',
        title='Whole-Body Human Pose Estimation in the Wild',
        container='Proceedings of the European '
        'Conference on Computer Vision (ECCV)',
        year='2020',
        homepage='https://github.com/jin-s13/COCO-WholeBody/',
    ),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        17:
        dict(
            name='left_big_toe',
            id=17,
            color=[255, 128, 0],
            type='lower',
            swap='right_big_toe'),
        18:
        dict(
            name='left_small_toe',
            id=18,
            color=[255, 128, 0],
            type='lower',
            swap='right_small_toe'),
        19:
        dict(
            name='left_heel',
            id=19,
            color=[255, 128, 0],
            type='lower',
            swap='right_heel'),
        20:
        dict(
            name='right_big_toe',
            id=20,
            color=[255, 128, 0],
            type='lower',
            swap='left_big_toe'),
        21:
        dict(
            name='right_small_toe',
            id=21,
            color=[255, 128, 0],
            type='lower',
            swap='left_small_toe'),
        22:
        dict(
            name='right_heel',
            id=22,
            color=[255, 128, 0],
            type='lower',
            swap='left_heel'),
        23:
        dict(
            name='face-0',
            id=23,
            color=[255, 255, 255],
            type='',
            swap='face-16'),
        24:
        dict(
            name='face-1',
            id=24,
            color=[255, 255, 255],
            type='',
            swap='face-15'),
        25:
        dict(
            name='face-2',
            id=25,
            color=[255, 255, 255],
            type='',
            swap='face-14'),
        26:
        dict(
            name='face-3',
            id=26,
            color=[255, 255, 255],
            type='',
            swap='face-13'),
        27:
        dict(
            name='face-4',
            id=27,
            color=[255, 255, 255],
            type='',
            swap='face-12'),
        28:
        dict(
            name='face-5',
            id=28,
            color=[255, 255, 255],
            type='',
            swap='face-11'),
        29:
        dict(
            name='face-6',
            id=29,
            color=[255, 255, 255],
            type='',
            swap='face-10'),
        30:
        dict(
            name='face-7',
            id=30,
            color=[255, 255, 255],
            type='',
            swap='face-9'),
        31:
        dict(name='face-8', id=31, color=[255, 255, 255], type='', swap=''),
        32:
        dict(
            name='face-9',
            id=32,
            color=[255, 255, 255],
            type='',
            swap='face-7'),
        33:
        dict(
            name='face-10',
            id=33,
            color=[255, 255, 255],
            type='',
            swap='face-6'),
        34:
        dict(
            name='face-11',
            id=34,
            color=[255, 255, 255],
            type='',
            swap='face-5'),
        35:
        dict(
            name='face-12',
            id=35,
            color=[255, 255, 255],
            type='',
            swap='face-4'),
        36:
        dict(
            name='face-13',
            id=36,
            color=[255, 255, 255],
            type='',
            swap='face-3'),
        37:
        dict(
            name='face-14',
            id=37,
            color=[255, 255, 255],
            type='',
            swap='face-2'),
        38:
        dict(
            name='face-15',
            id=38,
            color=[255, 255, 255],
            type='',
            swap='face-1'),
        39:
        dict(
            name='face-16',
            id=39,
            color=[255, 255, 255],
            type='',
            swap='face-0'),
        40:
        dict(
            name='face-17',
            id=40,
            color=[255, 255, 255],
            type='',
            swap='face-26'),
        41:
        dict(
            name='face-18',
            id=41,
            color=[255, 255, 255],
            type='',
            swap='face-25'),
        42:
        dict(
            name='face-19',
            id=42,
            color=[255, 255, 255],
            type='',
            swap='face-24'),
        43:
        dict(
            name='face-20',
            id=43,
            color=[255, 255, 255],
            type='',
            swap='face-23'),
        44:
        dict(
            name='face-21',
            id=44,
            color=[255, 255, 255],
            type='',
            swap='face-22'),
        45:
        dict(
            name='face-22',
            id=45,
            color=[255, 255, 255],
            type='',
            swap='face-21'),
        46:
        dict(
            name='face-23',
            id=46,
            color=[255, 255, 255],
            type='',
            swap='face-20'),
        47:
        dict(
            name='face-24',
            id=47,
            color=[255, 255, 255],
            type='',
            swap='face-19'),
        48:
        dict(
            name='face-25',
            id=48,
            color=[255, 255, 255],
            type='',
            swap='face-18'),
        49:
        dict(
            name='face-26',
            id=49,
            color=[255, 255, 255],
            type='',
            swap='face-17'),
        50:
        dict(name='face-27', id=50, color=[255, 255, 255], type='', swap=''),
        51:
        dict(name='face-28', id=51, color=[255, 255, 255], type='', swap=''),
        52:
        dict(name='face-29', id=52, color=[255, 255, 255], type='', swap=''),
        53:
        dict(name='face-30', id=53, color=[255, 255, 255], type='', swap=''),
        54:
        dict(
            name='face-31',
            id=54,
            color=[255, 255, 255],
            type='',
            swap='face-35'),
        55:
        dict(
            name='face-32',
            id=55,
            color=[255, 255, 255],
            type='',
            swap='face-34'),
        56:
        dict(name='face-33', id=56, color=[255, 255, 255], type='', swap=''),
        57:
        dict(
            name='face-34',
            id=57,
            color=[255, 255, 255],
            type='',
            swap='face-32'),
        58:
        dict(
            name='face-35',
            id=58,
            color=[255, 255, 255],
            type='',
            swap='face-31'),
        59:
        dict(
            name='face-36',
            id=59,
            color=[255, 255, 255],
            type='',
            swap='face-45'),
        60:
        dict(
            name='face-37',
            id=60,
            color=[255, 255, 255],
            type='',
            swap='face-44'),
        61:
        dict(
            name='face-38',
            id=61,
            color=[255, 255, 255],
            type='',
            swap='face-43'),
        62:
        dict(
            name='face-39',
            id=62,
            color=[255, 255, 255],
            type='',
            swap='face-42'),
        63:
        dict(
            name='face-40',
            id=63,
            color=[255, 255, 255],
            type='',
            swap='face-47'),
        64:
        dict(
            name='face-41',
            id=64,
            color=[255, 255, 255],
            type='',
            swap='face-46'),
        65:
        dict(
            name='face-42',
            id=65,
            color=[255, 255, 255],
            type='',
            swap='face-39'),
        66:
        dict(
            name='face-43',
            id=66,
            color=[255, 255, 255],
            type='',
            swap='face-38'),
        67:
        dict(
            name='face-44',
            id=67,
            color=[255, 255, 255],
            type='',
            swap='face-37'),
        68:
        dict(
            name='face-45',
            id=68,
            color=[255, 255, 255],
            type='',
            swap='face-36'),
        69:
        dict(
            name='face-46',
            id=69,
            color=[255, 255, 255],
            type='',
            swap='face-41'),
        70:
        dict(
            name='face-47',
            id=70,
            color=[255, 255, 255],
            type='',
            swap='face-40'),
        71:
        dict(
            name='face-48',
            id=71,
            color=[255, 255, 255],
            type='',
            swap='face-54'),
        72:
        dict(
            name='face-49',
            id=72,
            color=[255, 255, 255],
            type='',
            swap='face-53'),
        73:
        dict(
            name='face-50',
            id=73,
            color=[255, 255, 255],
            type='',
            swap='face-52'),
        74:
        dict(name='face-51', id=74, color=[255, 255, 255], type='', swap=''),
        75:
        dict(
            name='face-52',
            id=75,
            color=[255, 255, 255],
            type='',
            swap='face-50'),
        76:
        dict(
            name='face-53',
            id=76,
            color=[255, 255, 255],
            type='',
            swap='face-49'),
        77:
        dict(
            name='face-54',
            id=77,
            color=[255, 255, 255],
            type='',
            swap='face-48'),
        78:
        dict(
            name='face-55',
            id=78,
            color=[255, 255, 255],
            type='',
            swap='face-59'),
        79:
        dict(
            name='face-56',
            id=79,
            color=[255, 255, 255],
            type='',
            swap='face-58'),
        80:
        dict(name='face-57', id=80, color=[255, 255, 255], type='', swap=''),
        81:
        dict(
            name='face-58',
            id=81,
            color=[255, 255, 255],
            type='',
            swap='face-56'),
        82:
        dict(
            name='face-59',
            id=82,
            color=[255, 255, 255],
            type='',
            swap='face-55'),
        83:
        dict(
            name='face-60',
            id=83,
            color=[255, 255, 255],
            type='',
            swap='face-64'),
        84:
        dict(
            name='face-61',
            id=84,
            color=[255, 255, 255],
            type='',
            swap='face-63'),
        85:
        dict(name='face-62', id=85, color=[255, 255, 255], type='', swap=''),
        86:
        dict(
            name='face-63',
            id=86,
            color=[255, 255, 255],
            type='',
            swap='face-61'),
        87:
        dict(
            name='face-64',
            id=87,
            color=[255, 255, 255],
            type='',
            swap='face-60'),
        88:
        dict(
            name='face-65',
            id=88,
            color=[255, 255, 255],
            type='',
            swap='face-67'),
        89:
        dict(name='face-66', id=89, color=[255, 255, 255], type='', swap=''),
        90:
        dict(
            name='face-67',
            id=90,
            color=[255, 255, 255],
            type='',
            swap='face-65'),
        91:
        dict(
            name='left_hand_root',
            id=91,
            color=[255, 255, 255],
            type='',
            swap='right_hand_root'),
        92:
        dict(
            name='left_thumb1',
            id=92,
            color=[255, 128, 0],
            type='',
            swap='right_thumb1'),
        93:
        dict(
            name='left_thumb2',
            id=93,
            color=[255, 128, 0],
            type='',
            swap='right_thumb2'),
        94:
        dict(
            name='left_thumb3',
            id=94,
            color=[255, 128, 0],
            type='',
            swap='right_thumb3'),
        95:
        dict(
            name='left_thumb4',
            id=95,
            color=[255, 128, 0],
            type='',
            swap='right_thumb4'),
        96:
        dict(
            name='left_forefinger1',
            id=96,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger1'),
        97:
        dict(
            name='left_forefinger2',
            id=97,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger2'),
        98:
        dict(
            name='left_forefinger3',
            id=98,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger3'),
        99:
        dict(
            name='left_forefinger4',
            id=99,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger4'),
        100:
        dict(
            name='left_middle_finger1',
            id=100,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger1'),
        101:
        dict(
            name='left_middle_finger2',
            id=101,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger2'),
        102:
        dict(
            name='left_middle_finger3',
            id=102,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger3'),
        103:
        dict(
            name='left_middle_finger4',
            id=103,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger4'),
        104:
        dict(
            name='left_ring_finger1',
            id=104,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger1'),
        105:
        dict(
            name='left_ring_finger2',
            id=105,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger2'),
        106:
        dict(
            name='left_ring_finger3',
            id=106,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger3'),
        107:
        dict(
            name='left_ring_finger4',
            id=107,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger4'),
        108:
        dict(
            name='left_pinky_finger1',
            id=108,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger1'),
        109:
        dict(
            name='left_pinky_finger2',
            id=109,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger2'),
        110:
        dict(
            name='left_pinky_finger3',
            id=110,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger3'),
        111:
        dict(
            name='left_pinky_finger4',
            id=111,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger4'),
        112:
        dict(
            name='right_hand_root',
            id=112,
            color=[255, 255, 255],
            type='',
            swap='left_hand_root'),
        113:
        dict(
            name='right_thumb1',
            id=113,
            color=[255, 128, 0],
            type='',
            swap='left_thumb1'),
        114:
        dict(
            name='right_thumb2',
            id=114,
            color=[255, 128, 0],
            type='',
            swap='left_thumb2'),
        115:
        dict(
            name='right_thumb3',
            id=115,
            color=[255, 128, 0],
            type='',
            swap='left_thumb3'),
        116:
        dict(
            name='right_thumb4',
            id=116,
            color=[255, 128, 0],
            type='',
            swap='left_thumb4'),
        117:
        dict(
            name='right_forefinger1',
            id=117,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger1'),
        118:
        dict(
            name='right_forefinger2',
            id=118,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger2'),
        119:
        dict(
            name='right_forefinger3',
            id=119,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger3'),
        120:
        dict(
            name='right_forefinger4',
            id=120,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger4'),
        121:
        dict(
            name='right_middle_finger1',
            id=121,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger1'),
        122:
        dict(
            name='right_middle_finger2',
            id=122,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger2'),
        123:
        dict(
            name='right_middle_finger3',
            id=123,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger3'),
        124:
        dict(
            name='right_middle_finger4',
            id=124,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger4'),
        125:
        dict(
            name='right_ring_finger1',
            id=125,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger1'),
        126:
        dict(
            name='right_ring_finger2',
            id=126,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger2'),
        127:
        dict(
            name='right_ring_finger3',
            id=127,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger3'),
        128:
        dict(
            name='right_ring_finger4',
            id=128,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger4'),
        129:
        dict(
            name='right_pinky_finger1',
            id=129,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger1'),
        130:
        dict(
            name='right_pinky_finger2',
            id=130,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger2'),
        131:
        dict(
            name='right_pinky_finger3',
            id=131,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger3'),
        132:
        dict(
            name='right_pinky_finger4',
            id=132,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger4')
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        18:
        dict(
            link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255]),
        19:
        dict(link=('left_ankle', 'left_big_toe'), id=19, color=[0, 255, 0]),
        20:
        dict(link=('left_ankle', 'left_small_toe'), id=20, color=[0, 255, 0]),
        21:
        dict(link=('left_ankle', 'left_heel'), id=21, color=[0, 255, 0]),
        22:
        dict(
            link=('right_ankle', 'right_big_toe'), id=22, color=[255, 128, 0]),
        23:
        dict(
            link=('right_ankle', 'right_small_toe'),
            id=23,
            color=[255, 128, 0]),
        24:
        dict(link=('right_ankle', 'right_heel'), id=24, color=[255, 128, 0]),
        25:
        dict(
            link=('left_hand_root', 'left_thumb1'), id=25, color=[255, 128,
                                                                  0]),
        26:
        dict(link=('left_thumb1', 'left_thumb2'), id=26, color=[255, 128, 0]),
        27:
        dict(link=('left_thumb2', 'left_thumb3'), id=27, color=[255, 128, 0]),
        28:
        dict(link=('left_thumb3', 'left_thumb4'), id=28, color=[255, 128, 0]),
        29:
        dict(
            link=('left_hand_root', 'left_forefinger1'),
            id=29,
            color=[255, 153, 255]),
        30:
        dict(
            link=('left_forefinger1', 'left_forefinger2'),
            id=30,
            color=[255, 153, 255]),
        31:
        dict(
            link=('left_forefinger2', 'left_forefinger3'),
            id=31,
            color=[255, 153, 255]),
        32:
        dict(
            link=('left_forefinger3', 'left_forefinger4'),
            id=32,
            color=[255, 153, 255]),
        33:
        dict(
            link=('left_hand_root', 'left_middle_finger1'),
            id=33,
            color=[102, 178, 255]),
        34:
        dict(
            link=('left_middle_finger1', 'left_middle_finger2'),
            id=34,
            color=[102, 178, 255]),
        35:
        dict(
            link=('left_middle_finger2', 'left_middle_finger3'),
            id=35,
            color=[102, 178, 255]),
        36:
        dict(
            link=('left_middle_finger3', 'left_middle_finger4'),
            id=36,
            color=[102, 178, 255]),
        37:
        dict(
            link=('left_hand_root', 'left_ring_finger1'),
            id=37,
            color=[255, 51, 51]),
        38:
        dict(
            link=('left_ring_finger1', 'left_ring_finger2'),
            id=38,
            color=[255, 51, 51]),
        39:
        dict(
            link=('left_ring_finger2', 'left_ring_finger3'),
            id=39,
            color=[255, 51, 51]),
        40:
        dict(
            link=('left_ring_finger3', 'left_ring_finger4'),
            id=40,
            color=[255, 51, 51]),
        41:
        dict(
            link=('left_hand_root', 'left_pinky_finger1'),
            id=41,
            color=[0, 255, 0]),
        42:
        dict(
            link=('left_pinky_finger1', 'left_pinky_finger2'),
            id=42,
            color=[0, 255, 0]),
        43:
        dict(
            link=('left_pinky_finger2', 'left_pinky_finger3'),
            id=43,
            color=[0, 255, 0]),
        44:
        dict(
            link=('left_pinky_finger3', 'left_pinky_finger4'),
            id=44,
            color=[0, 255, 0]),
        45:
        dict(
            link=('right_hand_root', 'right_thumb1'),
            id=45,
            color=[255, 128, 0]),
        46:
        dict(
            link=('right_thumb1', 'right_thumb2'), id=46, color=[255, 128, 0]),
        47:
        dict(
            link=('right_thumb2', 'right_thumb3'), id=47, color=[255, 128, 0]),
        48:
        dict(
            link=('right_thumb3', 'right_thumb4'), id=48, color=[255, 128, 0]),
        49:
        dict(
            link=('right_hand_root', 'right_forefinger1'),
            id=49,
            color=[255, 153, 255]),
        50:
        dict(
            link=('right_forefinger1', 'right_forefinger2'),
            id=50,
            color=[255, 153, 255]),
        51:
        dict(
            link=('right_forefinger2', 'right_forefinger3'),
            id=51,
            color=[255, 153, 255]),
        52:
        dict(
            link=('right_forefinger3', 'right_forefinger4'),
            id=52,
            color=[255, 153, 255]),
        53:
        dict(
            link=('right_hand_root', 'right_middle_finger1'),
            id=53,
            color=[102, 178, 255]),
        54:
        dict(
            link=('right_middle_finger1', 'right_middle_finger2'),
            id=54,
            color=[102, 178, 255]),
        55:
        dict(
            link=('right_middle_finger2', 'right_middle_finger3'),
            id=55,
            color=[102, 178, 255]),
        56:
        dict(
            link=('right_middle_finger3', 'right_middle_finger4'),
            id=56,
            color=[102, 178, 255]),
        57:
        dict(
            link=('right_hand_root', 'right_ring_finger1'),
            id=57,
            color=[255, 51, 51]),
        58:
        dict(
            link=('right_ring_finger1', 'right_ring_finger2'),
            id=58,
            color=[255, 51, 51]),
        59:
        dict(
            link=('right_ring_finger2', 'right_ring_finger3'),
            id=59,
            color=[255, 51, 51]),
        60:
        dict(
            link=('right_ring_finger3', 'right_ring_finger4'),
            id=60,
            color=[255, 51, 51]),
        61:
        dict(
            link=('right_hand_root', 'right_pinky_finger1'),
            id=61,
            color=[0, 255, 0]),
        62:
        dict(
            link=('right_pinky_finger1', 'right_pinky_finger2'),
            id=62,
            color=[0, 255, 0]),
        63:
        dict(
            link=('right_pinky_finger2', 'right_pinky_finger3'),
            id=63,
            color=[0, 255, 0]),
        64:
        dict(
            link=('right_pinky_finger3', 'right_pinky_finger4'),
            id=64,
            color=[0, 255, 0])
    },
    joint_weights=[1.] * 133,
    # 'https://github.com/jin-s13/COCO-WholeBody/blob/master/'
    # 'evaluation/myeval_wholebody.py#L175'
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068, 0.066, 0.066,
        0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031,
        0.025, 0.020, 0.023, 0.029, 0.032, 0.037, 0.038, 0.043, 0.041, 0.045,
        0.013, 0.012, 0.011, 0.011, 0.012, 0.012, 0.011, 0.011, 0.013, 0.015,
        0.009, 0.007, 0.007, 0.007, 0.012, 0.009, 0.008, 0.016, 0.010, 0.017,
        0.011, 0.009, 0.011, 0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010,
        0.034, 0.008, 0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009,
        0.009, 0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01,
        0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035,
        0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02, 0.019,
        0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
        0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02,
        0.019, 0.022, 0.031
    ])

dataset_info = dict(
    dataset_name='goliath',
    paper_info=dict(
        author='',
        year='',
        homepage='',
    ),
    min_visible_keypoints=8,
    image_height=4096,
    image_width=2668,
    original_keypoint_info={
            0: 'nose',
            1: 'left_eye',
            2: 'right_eye',
            3: 'left_ear',
            4: 'right_ear',
            5: 'left_shoulder',
            6: 'right_shoulder',
            7: 'left_elbow',
            8: 'right_elbow',
            9: 'left_hip',
            10: 'right_hip',
            11: 'left_knee',
            12: 'right_knee',
            13: 'left_ankle',
            14: 'right_ankle',
            15: 'left_big_toe_tip',
            16: 'left_small_toe_tip',
            17: 'left_heel',
            18: 'right_big_toe_tip',
            19: 'right_small_toe_tip',
            20: 'right_heel',
            21: 'right_thumb_tip',
            22: 'right_thumb_first_joint',
            23: 'right_thumb_second_joint',
            24: 'right_thumb_third_joint',
            25: 'right_index_tip',
            26: 'right_index_first_joint',
            27: 'right_index_second_joint',
            28: 'right_index_third_joint',
            29: 'right_middle_tip',
            30: 'right_middle_first_joint',
            31: 'right_middle_second_joint',
            32: 'right_middle_third_joint',
            33: 'right_ring_tip',
            34: 'right_ring_first_joint',
            35: 'right_ring_second_joint',
            36: 'right_ring_third_joint',
            37: 'right_pinky_tip',
            38: 'right_pinky_first_joint',
            39: 'right_pinky_second_joint',
            40: 'right_pinky_third_joint',
            41: 'right_wrist',
            42: 'left_thumb_tip',
            43: 'left_thumb_first_joint',
            44: 'left_thumb_second_joint',
            45: 'left_thumb_third_joint',
            46: 'left_index_tip',
            47: 'left_index_first_joint',
            48: 'left_index_second_joint',
            49: 'left_index_third_joint',
            50: 'left_middle_tip',
            51: 'left_middle_first_joint',
            52: 'left_middle_second_joint',
            53: 'left_middle_third_joint',
            54: 'left_ring_tip',
            55: 'left_ring_first_joint',
            56: 'left_ring_second_joint',
            57: 'left_ring_third_joint',
            58: 'left_pinky_tip',
            59: 'left_pinky_first_joint',
            60: 'left_pinky_second_joint',
            61: 'left_pinky_third_joint',
            62: 'left_wrist',
            63: 'left_olecranon',
            64: 'right_olecranon',
            65: 'left_cubital_fossa',
            66: 'right_cubital_fossa',
            67: 'left_acromion',
            68: 'right_acromion',
            69: 'neck',
            70: 'center_of_glabella',
            71: 'center_of_nose_root',
            72: 'tip_of_nose_bridge',
            73: 'midpoint_1_of_nose_bridge',
            74: 'midpoint_2_of_nose_bridge',
            75: 'midpoint_3_of_nose_bridge',
            76: 'center_of_labiomental_groove',
            77: 'tip_of_chin',
            78: 'upper_startpoint_of_r_eyebrow',
            79: 'lower_startpoint_of_r_eyebrow',
            80: 'end_of_r_eyebrow',
            81: 'upper_midpoint_1_of_r_eyebrow',
            82: 'lower_midpoint_1_of_r_eyebrow',
            83: 'upper_midpoint_2_of_r_eyebrow',
            84: 'upper_midpoint_3_of_r_eyebrow',
            85: 'lower_midpoint_2_of_r_eyebrow',
            86: 'lower_midpoint_3_of_r_eyebrow',
            87: 'upper_startpoint_of_l_eyebrow',
            88: 'lower_startpoint_of_l_eyebrow',
            89: 'end_of_l_eyebrow',
            90: 'upper_midpoint_1_of_l_eyebrow',
            91: 'lower_midpoint_1_of_l_eyebrow',
            92: 'upper_midpoint_2_of_l_eyebrow',
            93: 'upper_midpoint_3_of_l_eyebrow',
            94: 'lower_midpoint_2_of_l_eyebrow',
            95: 'lower_midpoint_3_of_l_eyebrow',
            96: 'l_inner_end_of_upper_lash_line',
            97: 'l_outer_end_of_upper_lash_line',
            98: 'l_centerpoint_of_upper_lash_line',
            99: 'l_midpoint_2_of_upper_lash_line',
            100: 'l_midpoint_1_of_upper_lash_line',
            101: 'l_midpoint_6_of_upper_lash_line',
            102: 'l_midpoint_5_of_upper_lash_line',
            103: 'l_midpoint_4_of_upper_lash_line',
            104: 'l_midpoint_3_of_upper_lash_line',
            105: 'l_outer_end_of_upper_eyelid_line',
            106: 'l_midpoint_6_of_upper_eyelid_line',
            107: 'l_midpoint_2_of_upper_eyelid_line',
            108: 'l_midpoint_5_of_upper_eyelid_line',
            109: 'l_centerpoint_of_upper_eyelid_line',
            110: 'l_midpoint_4_of_upper_eyelid_line',
            111: 'l_midpoint_1_of_upper_eyelid_line',
            112: 'l_midpoint_3_of_upper_eyelid_line',
            113: 'l_midpoint_6_of_upper_crease_line',
            114: 'l_midpoint_2_of_upper_crease_line',
            115: 'l_midpoint_5_of_upper_crease_line',
            116: 'l_centerpoint_of_upper_crease_line',
            117: 'l_midpoint_4_of_upper_crease_line',
            118: 'l_midpoint_1_of_upper_crease_line',
            119: 'l_midpoint_3_of_upper_crease_line',
            120: 'r_inner_end_of_upper_lash_line',
            121: 'r_outer_end_of_upper_lash_line',
            122: 'r_centerpoint_of_upper_lash_line',
            123: 'r_midpoint_1_of_upper_lash_line',
            124: 'r_midpoint_2_of_upper_lash_line',
            125: 'r_midpoint_3_of_upper_lash_line',
            126: 'r_midpoint_4_of_upper_lash_line',
            127: 'r_midpoint_5_of_upper_lash_line',
            128: 'r_midpoint_6_of_upper_lash_line',
            129: 'r_outer_end_of_upper_eyelid_line',
            130: 'r_midpoint_3_of_upper_eyelid_line',
            131: 'r_midpoint_1_of_upper_eyelid_line',
            132: 'r_midpoint_4_of_upper_eyelid_line',
            133: 'r_centerpoint_of_upper_eyelid_line',
            134: 'r_midpoint_5_of_upper_eyelid_line',
            135: 'r_midpoint_2_of_upper_eyelid_line',
            136: 'r_midpoint_6_of_upper_eyelid_line',
            137: 'r_midpoint_3_of_upper_crease_line',
            138: 'r_midpoint_1_of_upper_crease_line',
            139: 'r_midpoint_4_of_upper_crease_line',
            140: 'r_centerpoint_of_upper_crease_line',
            141: 'r_midpoint_5_of_upper_crease_line',
            142: 'r_midpoint_2_of_upper_crease_line',
            143: 'r_midpoint_6_of_upper_crease_line',
            144: 'l_inner_end_of_lower_lash_line',
            145: 'l_outer_end_of_lower_lash_line',
            146: 'l_centerpoint_of_lower_lash_line',
            147: 'l_midpoint_2_of_lower_lash_line',
            148: 'l_midpoint_1_of_lower_lash_line',
            149: 'l_midpoint_6_of_lower_lash_line',
            150: 'l_midpoint_5_of_lower_lash_line',
            151: 'l_midpoint_4_of_lower_lash_line',
            152: 'l_midpoint_3_of_lower_lash_line',
            153: 'l_outer_end_of_lower_eyelid_line',
            154: 'l_midpoint_6_of_lower_eyelid_line',
            155: 'l_midpoint_2_of_lower_eyelid_line',
            156: 'l_midpoint_5_of_lower_eyelid_line',
            157: 'l_centerpoint_of_lower_eyelid_line',
            158: 'l_midpoint_4_of_lower_eyelid_line',
            159: 'l_midpoint_1_of_lower_eyelid_line',
            160: 'l_midpoint_3_of_lower_eyelid_line',
            161: 'r_inner_end_of_lower_lash_line',
            162: 'r_outer_end_of_lower_lash_line',
            163: 'r_centerpoint_of_lower_lash_line',
            164: 'r_midpoint_1_of_lower_lash_line',
            165: 'r_midpoint_2_of_lower_lash_line',
            166: 'r_midpoint_3_of_lower_lash_line',
            167: 'r_midpoint_4_of_lower_lash_line',
            168: 'r_midpoint_5_of_lower_lash_line',
            169: 'r_midpoint_6_of_lower_lash_line',
            170: 'r_outer_end_of_lower_eyelid_line',
            171: 'r_midpoint_3_of_lower_eyelid_line',
            172: 'r_midpoint_1_of_lower_eyelid_line',
            173: 'r_midpoint_4_of_lower_eyelid_line',
            174: 'r_centerpoint_of_lower_eyelid_line',
            175: 'r_midpoint_5_of_lower_eyelid_line',
            176: 'r_midpoint_2_of_lower_eyelid_line',
            177: 'r_midpoint_6_of_lower_eyelid_line',
            178: 'tip_of_nose',
            179: 'bottom_center_of_nose',
            180: 'r_outer_corner_of_nose',
            181: 'l_outer_corner_of_nose',
            182: 'inner_corner_of_r_nostril',
            183: 'outer_corner_of_r_nostril',
            184: 'upper_corner_of_r_nostril',
            185: 'inner_corner_of_l_nostril',
            186: 'outer_corner_of_l_nostril',
            187: 'upper_corner_of_l_nostril',
            188: 'r_outer_corner_of_mouth',
            189: 'l_outer_corner_of_mouth',
            190: 'center_of_cupid_bow',
            191: 'center_of_lower_outer_lip',
            192: 'midpoint_1_of_upper_outer_lip',
            193: 'midpoint_2_of_upper_outer_lip',
            194: 'midpoint_1_of_lower_outer_lip',
            195: 'midpoint_2_of_lower_outer_lip',
            196: 'midpoint_3_of_upper_outer_lip',
            197: 'midpoint_4_of_upper_outer_lip',
            198: 'midpoint_5_of_upper_outer_lip',
            199: 'midpoint_6_of_upper_outer_lip',
            200: 'midpoint_3_of_lower_outer_lip',
            201: 'midpoint_4_of_lower_outer_lip',
            202: 'midpoint_5_of_lower_outer_lip',
            203: 'midpoint_6_of_lower_outer_lip',
            204: 'r_inner_corner_of_mouth',
            205: 'l_inner_corner_of_mouth',
            206: 'center_of_upper_inner_lip',
            207: 'center_of_lower_inner_lip',
            208: 'midpoint_1_of_upper_inner_lip',
            209: 'midpoint_2_of_upper_inner_lip',
            210: 'midpoint_1_of_lower_inner_lip',
            211: 'midpoint_2_of_lower_inner_lip',
            212: 'midpoint_3_of_upper_inner_lip',
            213: 'midpoint_4_of_upper_inner_lip',
            214: 'midpoint_5_of_upper_inner_lip',
            215: 'midpoint_6_of_upper_inner_lip',
            216: 'midpoint_3_of_lower_inner_lip',
            217: 'midpoint_4_of_lower_inner_lip',
            218: 'midpoint_5_of_lower_inner_lip',
            219: 'midpoint_6_of_lower_inner_lip',
            220: 'teeth',
            221: 'teeth',
            222: 'teeth',
            223: 'teeth',
            224: 'teeth',
            225: 'teeth',
            226: 'teeth',
            227: 'teeth',
            228: 'teeth',
            229: 'teeth',
            230: 'teeth',
            231: 'teeth',
            232: 'teeth',
            233: 'teeth',
            234: 'teeth',
            235: 'teeth',
            236: 'teeth',
            237: 'teeth',
            238: 'teeth',
            239: 'teeth',
            240: 'teeth',
            241: 'teeth',
            242: 'teeth',
            243: 'teeth',
            244: 'teeth',
            245: 'teeth',
            246: 'teeth',
            247: 'teeth',
            248: 'teeth',
            249: 'teeth',
            250: 'teeth',
            251: 'teeth',
            252: 'teeth',
            253: 'teeth',
            254: 'teeth',
            255: 'teeth',
            256: 'l_top_end_of_inferior_crus',
            257: 'l_top_end_of_superior_crus',
            258: 'l_start_of_antihelix',
            259: 'l_end_of_antihelix',
            260: 'l_midpoint_1_of_antihelix',
            261: 'l_midpoint_1_of_inferior_crus',
            262: 'l_midpoint_2_of_antihelix',
            263: 'l_midpoint_3_of_antihelix',
            264: 'l_point_1_of_inner_helix',
            265: 'l_point_2_of_inner_helix',
            266: 'l_point_3_of_inner_helix',
            267: 'l_point_4_of_inner_helix',
            268: 'l_point_5_of_inner_helix',
            269: 'l_point_6_of_inner_helix',
            270: 'l_point_7_of_inner_helix',
            271: 'l_highest_point_of_antitragus',
            272: 'l_bottom_point_of_tragus',
            273: 'l_protruding_point_of_tragus',
            274: 'l_top_point_of_tragus',
            275: 'l_start_point_of_crus_of_helix',
            276: 'l_deepest_point_of_concha',
            277: 'l_tip_of_ear_lobe',
            278: 'l_midpoint_between_22_15',
            279: 'l_bottom_connecting_point_of_ear_lobe',
            280: 'l_top_connecting_point_of_helix',
            281: 'l_point_8_of_inner_helix',
            282: 'r_top_end_of_inferior_crus',
            283: 'r_top_end_of_superior_crus',
            284: 'r_start_of_antihelix',
            285: 'r_end_of_antihelix',
            286: 'r_midpoint_1_of_antihelix',
            287: 'r_midpoint_1_of_inferior_crus',
            288: 'r_midpoint_2_of_antihelix',
            289: 'r_midpoint_3_of_antihelix',
            290: 'r_point_1_of_inner_helix',
            291: 'r_point_8_of_inner_helix',
            292: 'r_point_3_of_inner_helix',
            293: 'r_point_4_of_inner_helix',
            294: 'r_point_5_of_inner_helix',
            295: 'r_point_6_of_inner_helix',
            296: 'r_point_7_of_inner_helix',
            297: 'r_highest_point_of_antitragus',
            298: 'r_bottom_point_of_tragus',
            299: 'r_protruding_point_of_tragus',
            300: 'r_top_point_of_tragus',
            301: 'r_start_point_of_crus_of_helix',
            302: 'r_deepest_point_of_concha',
            303: 'r_tip_of_ear_lobe',
            304: 'r_midpoint_between_22_15',
            305: 'r_bottom_connecting_point_of_ear_lobe',
            306: 'r_top_connecting_point_of_helix',
            307: 'r_point_2_of_inner_helix',
            308: 'l_center_of_iris',
            309: 'l_border_of_iris_3',
            310: 'l_border_of_iris_midpoint_1',
            311: 'l_border_of_iris_12',
            312: 'l_border_of_iris_midpoint_4',
            313: 'l_border_of_iris_9',
            314: 'l_border_of_iris_midpoint_3',
            315: 'l_border_of_iris_6',
            316: 'l_border_of_iris_midpoint_2',
            317: 'r_center_of_iris',
            318: 'r_border_of_iris_3',
            319: 'r_border_of_iris_midpoint_1',
            320: 'r_border_of_iris_12',
            321: 'r_border_of_iris_midpoint_4',
            322: 'r_border_of_iris_9',
            323: 'r_border_of_iris_midpoint_3',
            324: 'r_border_of_iris_6',
            325: 'r_border_of_iris_midpoint_2',
            326: 'l_center_of_pupil',
            327: 'l_border_of_pupil_3',
            328: 'l_border_of_pupil_midpoint_1',
            329: 'l_border_of_pupil_12',
            330: 'l_border_of_pupil_midpoint_4',
            331: 'l_border_of_pupil_9',
            332: 'l_border_of_pupil_midpoint_3',
            333: 'l_border_of_pupil_6',
            334: 'l_border_of_pupil_midpoint_2',
            335: 'r_center_of_pupil',
            336: 'r_border_of_pupil_3',
            337: 'r_border_of_pupil_midpoint_1',
            338: 'r_border_of_pupil_12',
            339: 'r_border_of_pupil_midpoint_4',
            340: 'r_border_of_pupil_9',
            341: 'r_border_of_pupil_midpoint_3',
            342: 'r_border_of_pupil_6',
            343: 'r_border_of_pupil_midpoint_2',
            },
    keypoint_info={
        0: dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1: dict(name='left_eye', id=1, color=[51, 153, 255], type='upper', swap='right_eye'),
        2: dict(name='right_eye', id=2, color=[51, 153, 255], type='upper', swap='left_eye'),
        3: dict(name='left_ear', id=3, color=[51, 153, 255], type='upper', swap='right_ear'),
        4: dict(name='right_ear', id=4, color=[51, 153, 255], type='upper', swap='left_ear'),
        5: dict(name='left_shoulder', id=5, color=[51, 153, 255], type='upper', swap='right_shoulder'),
        6: dict(name='right_shoulder', id=6, color=[51, 153, 255], type='upper', swap='left_shoulder'),
        7: dict(name='left_elbow', id=7, color=[51, 153, 255], type='upper', swap='right_elbow'),
        8: dict(name='right_elbow', id=8, color=[51, 153, 255], type='upper', swap='left_elbow'),
        9: dict(name='left_hip', id=9, color=[51, 153, 255], type='lower', swap='right_hip'),
        10: dict(name='right_hip', id=10, color=[51, 153, 255], type='lower', swap='left_hip'),
        11: dict(name='left_knee', id=11, color=[51, 153, 255], type='lower', swap='right_knee'),
        12: dict(name='right_knee', id=12, color=[51, 153, 255], type='lower', swap='left_knee'),
        13: dict(name='left_ankle', id=13, color=[51, 153, 255], type='lower', swap='right_ankle'),
        14: dict(name='right_ankle', id=14, color=[51, 153, 255], type='lower', swap='left_ankle'),

        15: dict(name='left_big_toe', id=15, color=[51, 153, 255], type='lower', swap='right_big_toe'),
        16: dict(name='left_small_toe', id=16, color=[51, 153, 255], type='lower', swap='right_small_toe'),
        17: dict(name='left_heel', id=17, color=[51, 153, 255], type='lower', swap='right_heel'),
        18: dict(name='right_big_toe', id=18, color=[51, 153, 255], type='lower', swap='left_big_toe'),
        19: dict(name='right_small_toe', id=19, color=[51, 153, 255], type='lower', swap='left_small_toe'),
        20: dict(name='right_heel', id=20, color=[51, 153, 255], type='lower', swap='left_heel'),

        21: dict(name='right_thumb4', id=21, color=[51, 153, 255], type='upper', swap='left_thumb4'),
        22: dict(name='right_thumb3', id=22, color=[51, 153, 255], type='upper', swap='left_thumb3'),
        23: dict(name='right_thumb2', id=23, color=[51, 153, 255], type='upper', swap='left_thumb2'),
        24: dict(name='right_thumb_third_joint', id=24, color=[51, 153, 255], type='upper', swap='left_thumb_third_joint'),

        25: dict(name='right_forefinger4', id=25, color=[51, 153, 255], type='upper', swap='left_forefinger4'),
        26: dict(name='right_forefinger3', id=26, color=[51, 153, 255], type='upper', swap='left_forefinger3'),
        27: dict(name='right_forefinger2', id=27, color=[51, 153, 255], type='upper', swap='left_forefinger2'),
        28: dict(name='right_forefinger_third_joint', id=28, color=[51, 153, 255], type='upper', swap='left_forefinger_third_joint'),

        29: dict(name='right_middle_finger4', id=29, color=[51, 153, 255], type='upper', swap='left_middle_finger4'),
        30: dict(name='right_middle_finger3', id=30, color=[51, 153, 255], type='upper', swap='left_middle_finger3'),
        31: dict(name='right_middle_finger2', id=31, color=[51, 153, 255], type='upper', swap='left_middle_finger2'),
        32: dict(name='right_middle_finger_third_joint', id=32, color=[51, 153, 255], type='upper', swap='left_middle_finger_third_joint'),

        33: dict(name='right_ring_finger4', id=33, color=[51, 153, 255], type='upper', swap='left_ring_finger4'),
        34: dict(name='right_ring_finger3', id=34, color=[51, 153, 255], type='upper', swap='left_ring_finger3'),
        35: dict(name='right_ring_finger2', id=35, color=[51, 153, 255], type='upper', swap='left_ring_finger2'),
        36: dict(name='right_ring_finger_third_joint', id=36, color=[51, 153, 255], type='upper', swap='left_ring_finger_third_joint'),

        37: dict(name='right_pinky_finger4', id=37, color=[51, 153, 255], type='upper', swap='left_pinky_finger4'),
        38: dict(name='right_pinky_finger3', id=38, color=[51, 153, 255], type='upper', swap='left_pinky_finger3'),
        39: dict(name='right_pinky_finger2', id=39, color=[51, 153, 255], type='upper', swap='left_pinky_finger2'),
        40: dict(name='right_pinky_finger_third_joint', id=40, color=[51, 153, 255], type='upper', swap='left_pinky_finger_third_joint'),

        41: dict(name='right_wrist', id=41, color=[51, 153, 255], type='upper', swap='left_wrist'),

        42: dict(name='left_thumb4', id=42, color=[51, 153, 255], type='upper', swap='right_thumb4'),
        43: dict(name='left_thumb3', id=43, color=[51, 153, 255], type='upper', swap='right_thumb3'),
        44: dict(name='left_thumb2', id=44, color=[51, 153, 255], type='upper', swap='right_thumb2'),
        45: dict(name='left_thumb_third_joint', id=45, color=[51, 153, 255], type='upper', swap='right_thumb_third_joint'), ## doesnt match with wholebody

        46: dict(name='left_forefinger4', id=46, color=[51, 153, 255], type='upper', swap='right_forefinger4'),
        47: dict(name='left_forefinger3', id=47, color=[51, 153, 255], type='upper', swap='right_forefinger3'),
        48: dict(name='left_forefinger2', id=48, color=[51, 153, 255], type='upper', swap='right_forefinger2'),
        49: dict(name='left_forefinger_third_joint', id=49, color=[51, 153, 255], type='upper', swap='right_forefinger_third_joint'),

        50: dict(name='left_middle_finger4', id=50, color=[51, 153, 255], type='upper', swap='right_middle_finger4'),
        51: dict(name='left_middle_finger3', id=51, color=[51, 153, 255], type='upper', swap='right_middle_finger3'),
        52: dict(name='left_middle_finger2', id=52, color=[51, 153, 255], type='upper', swap='right_middle_finger2'),
        53: dict(name='left_middle_finger_third_joint', id=53, color=[51, 153, 255], type='upper', swap='right_middle_finger_third_joint'),

        54: dict(name='left_ring_finger4', id=54, color=[51, 153, 255], type='upper', swap='right_ring_finger4'),
        55: dict(name='left_ring_finger3', id=55, color=[51, 153, 255], type='upper', swap='right_ring_finger3'),
        56: dict(name='left_ring_finger2', id=56, color=[51, 153, 255], type='upper', swap='right_ring_finger2'),
        57: dict(name='left_ring_finger_third_joint', id=57, color=[51, 153, 255], type='upper', swap='right_ring_finger_third_joint'),

        58: dict(name='left_pinky_finger4', id=58, color=[51, 153, 255], type='upper', swap='right_pinky_finger4'),
        59: dict(name='left_pinky_finger3', id=59, color=[51, 153, 255], type='upper', swap='right_pinky_finger3'),
        60: dict(name='left_pinky_finger2', id=60, color=[51, 153, 255], type='upper', swap='right_pinky_finger2'),
        61: dict(name='left_pinky_finger_third_joint', id=61, color=[51, 153, 255], type='upper', swap='right_pinky_finger_third_joint'),

        62: dict(name='left_wrist', id=62, color=[51, 153, 255], type='upper', swap='right_wrist'),

        63: dict(name='left_olecranon', id=63, color=[51, 153, 255], type='', swap='right_olecranon'),
        64: dict(name='right_olecranon', id=64, color=[51, 153, 255], type='', swap='left_olecranon'),
        65: dict(name='left_cubital_fossa', id=65, color=[51, 153, 255], type='', swap='right_cubital_fossa'),
        66: dict(name='right_cubital_fossa', id=66, color=[51, 153, 255], type='', swap='left_cubital_fossa'),
        67: dict(name='left_acromion', id=67, color=[51, 153, 255], type='', swap='right_acromion'),
        68: dict(name='right_acromion', id=68, color=[51, 153, 255], type='', swap='left_acromion'),
        69: dict(name='neck', id=69, color=[51, 153, 255], type='', swap=''),

        70: dict(name='center_of_glabella', id=70, color=[255, 255, 255], type='', swap=''),
        71: dict(name='center_of_nose_root', id=71, color=[255, 255, 255], type='', swap=''),
        72: dict(name='tip_of_nose_bridge', id=72, color=[255, 255, 255], type='', swap=''),
        73: dict(name='midpoint_1_of_nose_bridge', id=73, color=[255, 255, 255], type='', swap=''),
        74: dict(name='midpoint_2_of_nose_bridge', id=74, color=[255, 255, 255], type='', swap=''),
        75: dict(name='midpoint_3_of_nose_bridge', id=75, color=[255, 255, 255], type='', swap=''),
        76: dict(name='center_of_labiomental_groove', id=76, color=[255, 255, 255], type='', swap=''),
        77: dict(name='tip_of_chin', id=77, color=[255, 255, 255], type='', swap=''),
        78: dict(name='upper_startpoint_of_r_eyebrow', id=78, color=[255, 255, 255], type='', swap='upper_startpoint_of_l_eyebrow'),
        79: dict(name='lower_startpoint_of_r_eyebrow', id=79, color=[255, 255, 255], type='', swap='lower_startpoint_of_l_eyebrow'),
        80: dict(name='end_of_r_eyebrow', id=80, color=[255, 255, 255], type='', swap='end_of_l_eyebrow'),
        81: dict(name='upper_midpoint_1_of_r_eyebrow', id=81, color=[255, 255, 255], type='', swap='upper_midpoint_1_of_l_eyebrow'),
        82: dict(name='lower_midpoint_1_of_r_eyebrow', id=82, color=[255, 255, 255], type='', swap='lower_midpoint_1_of_l_eyebrow'),
        83: dict(name='upper_midpoint_2_of_r_eyebrow', id=83, color=[255, 255, 255], type='', swap='upper_midpoint_3_of_l_eyebrow'),
        84: dict(name='upper_midpoint_3_of_r_eyebrow', id=84, color=[255, 255, 255], type='', swap='upper_midpoint_2_of_l_eyebrow'),
        85: dict(name='lower_midpoint_2_of_r_eyebrow', id=85, color=[255, 255, 255], type='', swap='lower_midpoint_3_of_l_eyebrow'),
        86: dict(name='lower_midpoint_3_of_r_eyebrow', id=86, color=[255, 255, 255], type='', swap='lower_midpoint_2_of_l_eyebrow'),
        87: dict(name='upper_startpoint_of_l_eyebrow', id=87, color=[255, 255, 255], type='', swap='upper_startpoint_of_r_eyebrow'),
        88: dict(name='lower_startpoint_of_l_eyebrow', id=88, color=[255, 255, 255], type='', swap='lower_startpoint_of_r_eyebrow'),
        89: dict(name='end_of_l_eyebrow', id=89, color=[255, 255, 255], type='', swap='end_of_r_eyebrow'),
        90: dict(name='upper_midpoint_1_of_l_eyebrow', id=90, color=[255, 255, 255], type='', swap='upper_midpoint_1_of_r_eyebrow'),
        91: dict(name='lower_midpoint_1_of_l_eyebrow', id=91, color=[255, 255, 255], type='', swap='lower_midpoint_1_of_r_eyebrow'),
        92: dict(name='upper_midpoint_2_of_l_eyebrow', id=92, color=[255, 255, 255], type='', swap='upper_midpoint_3_of_r_eyebrow'),
        93: dict(name='upper_midpoint_3_of_l_eyebrow', id=93, color=[255, 255, 255], type='', swap='upper_midpoint_2_of_r_eyebrow'),
        94: dict(name='lower_midpoint_2_of_l_eyebrow', id=94, color=[255, 255, 255], type='', swap='lower_midpoint_3_of_r_eyebrow'),
        95: dict(name='lower_midpoint_3_of_l_eyebrow', id=95, color=[255, 255, 255], type='', swap='lower_midpoint_2_of_r_eyebrow'),

        96: dict(name='l_inner_end_of_upper_lash_line', id=96, color=[192, 64, 128], type='', swap='r_inner_end_of_upper_lash_line'),
        97: dict(name='l_outer_end_of_upper_lash_line', id=97, color=[192, 64, 128], type='', swap='r_outer_end_of_upper_lash_line'),
        98: dict(name='l_centerpoint_of_upper_lash_line', id=98, color=[192, 64, 128], type='', swap='r_centerpoint_of_upper_lash_line'),
        99: dict(name='l_midpoint_2_of_upper_lash_line', id=99, color=[192, 64, 128], type='', swap='r_midpoint_1_of_upper_lash_line'),
        100: dict(name='l_midpoint_1_of_upper_lash_line', id=100, color=[192, 64, 128], type='', swap='r_midpoint_2_of_upper_lash_line'),
        101: dict(name='l_midpoint_6_of_upper_lash_line', id=101, color=[192, 64, 128], type='', swap='r_midpoint_3_of_upper_lash_line'),
        102: dict(name='l_midpoint_5_of_upper_lash_line', id=102, color=[192, 64, 128], type='', swap='r_midpoint_4_of_upper_lash_line'),
        103: dict(name='l_midpoint_4_of_upper_lash_line', id=103, color=[192, 64, 128], type='', swap='r_midpoint_5_of_upper_lash_line'),
        104: dict(name='l_midpoint_3_of_upper_lash_line', id=104, color=[192, 64, 128], type='', swap='r_midpoint_6_of_upper_lash_line'),

        105: dict(name='l_outer_end_of_upper_eyelid_line', id=105, color=[192, 64, 128], type='', swap='r_outer_end_of_upper_eyelid_line'),
        106: dict(name='l_midpoint_6_of_upper_eyelid_line', id=106, color=[192, 64, 128], type='', swap='r_midpoint_3_of_upper_eyelid_line'),
        107: dict(name='l_midpoint_2_of_upper_eyelid_line', id=107, color=[192, 64, 128], type='', swap='r_midpoint_1_of_upper_eyelid_line'),
        108: dict(name='l_midpoint_5_of_upper_eyelid_line', id=108, color=[192, 64, 128], type='', swap='r_midpoint_4_of_upper_eyelid_line'),
        109: dict(name='l_centerpoint_of_upper_eyelid_line', id=109, color=[192, 64, 128], type='', swap='r_centerpoint_of_upper_eyelid_line'),
        110: dict(name='l_midpoint_4_of_upper_eyelid_line', id=110, color=[192, 64, 128], type='', swap='r_midpoint_5_of_upper_eyelid_line'),
        111: dict(name='l_midpoint_1_of_upper_eyelid_line', id=111, color=[192, 64, 128], type='', swap='r_midpoint_2_of_upper_eyelid_line'),
        112: dict(name='l_midpoint_3_of_upper_eyelid_line', id=112, color=[192, 64, 128], type='', swap='r_midpoint_6_of_upper_eyelid_line'),
        113: dict(name='l_midpoint_6_of_upper_crease_line', id=113, color=[192, 64, 128], type='', swap='r_midpoint_3_of_upper_crease_line'),

        114: dict(name='l_midpoint_2_of_upper_crease_line', id=114, color=[192, 64, 128], type='', swap='r_midpoint_1_of_upper_crease_line'),
        115: dict(name='l_midpoint_5_of_upper_crease_line', id=115, color=[192, 64, 128], type='', swap='r_midpoint_4_of_upper_crease_line'),
        116: dict(name='l_centerpoint_of_upper_crease_line', id=116, color=[192, 64, 128], type='', swap='r_centerpoint_of_upper_crease_line'),
        117: dict(name='l_midpoint_4_of_upper_crease_line', id=117, color=[192, 64, 128], type='', swap='r_midpoint_5_of_upper_crease_line'),
        118: dict(name='l_midpoint_1_of_upper_crease_line', id=118, color=[192, 64, 128], type='', swap='r_midpoint_2_of_upper_crease_line'),
        119: dict(name='l_midpoint_3_of_upper_crease_line', id=119, color=[192, 64, 128], type='', swap='r_midpoint_6_of_upper_crease_line'),

        120: dict(name='r_inner_end_of_upper_lash_line', id=120, color=[64, 32, 192], type='', swap='l_inner_end_of_upper_lash_line'),
        121: dict(name='r_outer_end_of_upper_lash_line', id=121, color=[64, 32, 192], type='', swap='l_outer_end_of_upper_lash_line'),
        122: dict(name='r_centerpoint_of_upper_lash_line', id=122, color=[64, 32, 192], type='', swap='l_centerpoint_of_upper_lash_line'),
        123: dict(name='r_midpoint_1_of_upper_lash_line', id=123, color=[64, 32, 192], type='', swap='l_midpoint_2_of_upper_lash_line'),
        124: dict(name='r_midpoint_2_of_upper_lash_line', id=124, color=[64, 32, 192], type='', swap='l_midpoint_1_of_upper_lash_line'),
        125: dict(name='r_midpoint_3_of_upper_lash_line', id=125, color=[64, 32, 192], type='', swap='l_midpoint_6_of_upper_lash_line'),
        126: dict(name='r_midpoint_4_of_upper_lash_line', id=126, color=[64, 32, 192], type='', swap='l_midpoint_5_of_upper_lash_line'),
        127: dict(name='r_midpoint_5_of_upper_lash_line', id=127, color=[64, 32, 192], type='', swap='l_midpoint_4_of_upper_lash_line'),

        128: dict(name='r_midpoint_6_of_upper_lash_line', id=128, color=[64, 32, 192], type='', swap='l_midpoint_3_of_upper_lash_line'),
        129: dict(name='r_outer_end_of_upper_eyelid_line', id=129, color=[64, 32, 192], type='', swap='l_outer_end_of_upper_eyelid_line'),
        130: dict(name='r_midpoint_3_of_upper_eyelid_line', id=130, color=[64, 32, 192], type='', swap='l_midpoint_6_of_upper_eyelid_line'),
        131: dict(name='r_midpoint_1_of_upper_eyelid_line', id=131, color=[64, 32, 192], type='', swap='l_midpoint_2_of_upper_eyelid_line'),
        132: dict(name='r_midpoint_4_of_upper_eyelid_line', id=132, color=[64, 32, 192], type='', swap='l_midpoint_5_of_upper_eyelid_line'),
        133: dict(name='r_centerpoint_of_upper_eyelid_line', id=133, color=[64, 32, 192], type='', swap='l_centerpoint_of_upper_eyelid_line'),
        134: dict(name='r_midpoint_5_of_upper_eyelid_line', id=134, color=[64, 32, 192], type='', swap='l_midpoint_4_of_upper_eyelid_line'),
        135: dict(name='r_midpoint_2_of_upper_eyelid_line', id=135, color=[64, 32, 192], type='', swap='l_midpoint_1_of_upper_eyelid_line'),

        136: dict(name='r_midpoint_6_of_upper_eyelid_line', id=136, color=[64, 32, 192], type='', swap='l_midpoint_3_of_upper_eyelid_line'),
        137: dict(name='r_midpoint_3_of_upper_crease_line', id=137, color=[64, 32, 192], type='', swap='l_midpoint_6_of_upper_crease_line'),
        138: dict(name='r_midpoint_1_of_upper_crease_line', id=138, color=[64, 32, 192], type='', swap='l_midpoint_2_of_upper_crease_line'),
        139: dict(name='r_midpoint_4_of_upper_crease_line', id=139, color=[64, 32, 192], type='', swap='l_midpoint_5_of_upper_crease_line'),
        140: dict(name='r_centerpoint_of_upper_crease_line', id=140, color=[64, 32, 192], type='', swap='l_centerpoint_of_upper_crease_line'),
        141: dict(name='r_midpoint_5_of_upper_crease_line', id=141, color=[64, 32, 192], type='', swap='l_midpoint_4_of_upper_crease_line'),
        142: dict(name='r_midpoint_2_of_upper_crease_line', id=142, color=[64, 32, 192], type='', swap='l_midpoint_1_of_upper_crease_line'),
        143: dict(name='r_midpoint_6_of_upper_crease_line', id=143, color=[64, 32, 192], type='', swap='l_midpoint_3_of_upper_crease_line'),

        144: dict(name='l_inner_end_of_lower_lash_line', id=144, color=[64, 192, 128], type='', swap='r_inner_end_of_lower_lash_line'),
        145: dict(name='l_outer_end_of_lower_lash_line', id=145, color=[64, 192, 128], type='', swap='r_outer_end_of_lower_lash_line'),
        146: dict(name='l_centerpoint_of_lower_lash_line', id=146, color=[64, 192, 128], type='', swap='r_centerpoint_of_lower_lash_line'),
        147: dict(name='l_midpoint_2_of_lower_lash_line', id=147, color=[64, 192, 128], type='', swap='r_midpoint_1_of_lower_lash_line'),
        148: dict(name='l_midpoint_1_of_lower_lash_line', id=148, color=[64, 192, 128], type='', swap='r_midpoint_2_of_lower_lash_line'),
        149: dict(name='l_midpoint_6_of_lower_lash_line', id=149, color=[64, 192, 128], type='', swap='r_midpoint_3_of_lower_lash_line'),
        150: dict(name='l_midpoint_5_of_lower_lash_line', id=150, color=[64, 192, 128], type='', swap='r_midpoint_4_of_lower_lash_line'),
        151: dict(name='l_midpoint_4_of_lower_lash_line', id=151, color=[64, 192, 128], type='', swap='r_midpoint_5_of_lower_lash_line'),
        152: dict(name='l_midpoint_3_of_lower_lash_line', id=152, color=[64, 192, 128], type='', swap='r_midpoint_6_of_lower_lash_line'),
        153: dict(name='l_outer_end_of_lower_eyelid_line', id=153, color=[64, 192, 128], type='', swap='r_outer_end_of_lower_eyelid_line'),
        154: dict(name='l_midpoint_6_of_lower_eyelid_line', id=154, color=[64, 192, 128], type='', swap='r_midpoint_3_of_lower_eyelid_line'),
        155: dict(name='l_midpoint_2_of_lower_eyelid_line', id=155, color=[64, 192, 128], type='', swap='r_midpoint_1_of_lower_eyelid_line'),
        156: dict(name='l_midpoint_5_of_lower_eyelid_line', id=156, color=[64, 192, 128], type='', swap='r_midpoint_4_of_lower_eyelid_line'),
        157: dict(name='l_centerpoint_of_lower_eyelid_line', id=157, color=[64, 192, 128], type='', swap='r_centerpoint_of_lower_eyelid_line'),
        158: dict(name='l_midpoint_4_of_lower_eyelid_line', id=158, color=[64, 192, 128], type='', swap='r_midpoint_5_of_lower_eyelid_line'),
        159: dict(name='l_midpoint_1_of_lower_eyelid_line', id=159, color=[64, 192, 128], type='', swap='r_midpoint_2_of_lower_eyelid_line'),
        160: dict(name='l_midpoint_3_of_lower_eyelid_line', id=160, color=[64, 192, 128], type='', swap='r_midpoint_6_of_lower_eyelid_line'),

        161: dict(name='r_inner_end_of_lower_lash_line', id=161, color=[64, 192, 32], type='', swap='l_inner_end_of_lower_lash_line'),
        162: dict(name='r_outer_end_of_lower_lash_line', id=162, color=[64, 192, 32], type='', swap='l_outer_end_of_lower_lash_line'),
        163: dict(name='r_centerpoint_of_lower_lash_line', id=163, color=[64, 192, 32], type='', swap='l_centerpoint_of_lower_lash_line'),
        164: dict(name='r_midpoint_1_of_lower_lash_line', id=164, color=[64, 192, 32], type='', swap='l_midpoint_2_of_lower_lash_line'),
        165: dict(name='r_midpoint_2_of_lower_lash_line', id=165, color=[64, 192, 32], type='', swap='l_midpoint_1_of_lower_lash_line'),
        166: dict(name='r_midpoint_3_of_lower_lash_line', id=166, color=[64, 192, 32], type='', swap='l_midpoint_6_of_lower_lash_line'),
        167: dict(name='r_midpoint_4_of_lower_lash_line', id=167, color=[64, 192, 32], type='', swap='l_midpoint_5_of_lower_lash_line'),
        168: dict(name='r_midpoint_5_of_lower_lash_line', id=168, color=[64, 192, 32], type='', swap='l_midpoint_4_of_lower_lash_line'),
        169: dict(name='r_midpoint_6_of_lower_lash_line', id=169, color=[64, 192, 32], type='', swap='l_midpoint_3_of_lower_lash_line'),
        170: dict(name='r_outer_end_of_lower_eyelid_line', id=170, color=[64, 192, 32], type='', swap='l_outer_end_of_lower_eyelid_line'),
        171: dict(name='r_midpoint_3_of_lower_eyelid_line', id=171, color=[64, 192, 32], type='', swap='l_midpoint_6_of_lower_eyelid_line'),
        172: dict(name='r_midpoint_1_of_lower_eyelid_line', id=172, color=[64, 192, 32], type='', swap='l_midpoint_2_of_lower_eyelid_line'),
        173: dict(name='r_midpoint_4_of_lower_eyelid_line', id=173, color=[64, 192, 32], type='', swap='l_midpoint_5_of_lower_eyelid_line'),
        174: dict(name='r_centerpoint_of_lower_eyelid_line', id=174, color=[64, 192, 32], type='', swap='l_centerpoint_of_lower_eyelid_line'),
        175: dict(name='r_midpoint_5_of_lower_eyelid_line', id=175, color=[64, 192, 32], type='', swap='l_midpoint_4_of_lower_eyelid_line'),
        176: dict(name='r_midpoint_2_of_lower_eyelid_line', id=176, color=[64, 192, 32], type='', swap='l_midpoint_1_of_lower_eyelid_line'),
        177: dict(name='r_midpoint_6_of_lower_eyelid_line', id=177, color=[64, 192, 32], type='', swap='l_midpoint_3_of_lower_eyelid_line'),

        178: dict(name='tip_of_nose', id=178, color=[0, 192, 0], type='', swap=''),
        179: dict(name='bottom_center_of_nose', id=179, color=[0, 192, 0], type='', swap=''),
        180: dict(name='r_outer_corner_of_nose', id=180, color=[0, 192, 0], type='', swap='l_outer_corner_of_nose'),
        181: dict(name='l_outer_corner_of_nose', id=181, color=[0, 192, 0], type='', swap='r_outer_corner_of_nose'),
        182: dict(name='inner_corner_of_r_nostril', id=182, color=[0, 192, 0], type='', swap='inner_corner_of_l_nostril'),
        183: dict(name='outer_corner_of_r_nostril', id=183, color=[0, 192, 0], type='', swap='outer_corner_of_l_nostril'),
        184: dict(name='upper_corner_of_r_nostril', id=184, color=[0, 192, 0], type='', swap='upper_corner_of_l_nostril'),
        185: dict(name='inner_corner_of_l_nostril', id=185, color=[0, 192, 0], type='', swap='inner_corner_of_r_nostril'),
        186: dict(name='outer_corner_of_l_nostril', id=186, color=[0, 192, 0], type='', swap='outer_corner_of_r_nostril'),
        187: dict(name='upper_corner_of_l_nostril', id=187, color=[0, 192, 0], type='', swap='upper_corner_of_r_nostril'),

        188: dict(name='r_outer_corner_of_mouth', id=188, color=[192, 0, 0], type='', swap='l_outer_corner_of_mouth'),
        189: dict(name='l_outer_corner_of_mouth', id=189, color=[192, 0, 0], type='', swap='r_outer_corner_of_mouth'),
        190: dict(name='center_of_cupid_bow', id=190, color=[192, 0, 0], type='', swap=''),
        191: dict(name='center_of_lower_outer_lip', id=191, color=[192, 0, 0], type='', swap=''),
        192: dict(name='midpoint_1_of_upper_outer_lip', id=192, color=[192, 0, 0], type='', swap='midpoint_2_of_upper_outer_lip'),
        193: dict(name='midpoint_2_of_upper_outer_lip', id=193, color=[192, 0, 0], type='', swap='midpoint_1_of_upper_outer_lip'),
        194: dict(name='midpoint_1_of_lower_outer_lip', id=194, color=[192, 0, 0], type='', swap='midpoint_2_of_lower_outer_lip'),
        195: dict(name='midpoint_2_of_lower_outer_lip', id=195, color=[192, 0, 0], type='', swap='midpoint_1_of_lower_outer_lip'),

        196: dict(name='midpoint_3_of_upper_outer_lip', id=196, color=[192, 0, 0], type='', swap='midpoint_6_of_upper_outer_lip'),
        197: dict(name='midpoint_4_of_upper_outer_lip', id=197, color=[192, 0, 0], type='', swap='midpoint_5_of_upper_outer_lip'),
        198: dict(name='midpoint_5_of_upper_outer_lip', id=198, color=[192, 0, 0], type='', swap='midpoint_4_of_upper_outer_lip'),
        199: dict(name='midpoint_6_of_upper_outer_lip', id=199, color=[192, 0, 0], type='', swap='midpoint_3_of_upper_outer_lip'),
        200: dict(name='midpoint_3_of_lower_outer_lip', id=200, color=[192, 0, 0], type='', swap='midpoint_6_of_lower_outer_lip'),
        201: dict(name='midpoint_4_of_lower_outer_lip', id=201, color=[192, 0, 0], type='', swap='midpoint_5_of_lower_outer_lip'),
        202: dict(name='midpoint_5_of_lower_outer_lip', id=202, color=[192, 0, 0], type='', swap='midpoint_4_of_lower_outer_lip'),
        203: dict(name='midpoint_6_of_lower_outer_lip', id=203, color=[192, 0, 0], type='', swap='midpoint_3_of_lower_outer_lip'),

        204: dict(name='r_inner_corner_of_mouth', id=204, color=[0, 192, 192], type='', swap='l_inner_corner_of_mouth'),
        205: dict(name='l_inner_corner_of_mouth', id=205, color=[0, 192, 192], type='', swap='r_inner_corner_of_mouth'),
        206: dict(name='center_of_upper_inner_lip', id=206, color=[0, 192, 192], type='', swap=''),
        207: dict(name='center_of_lower_inner_lip', id=207, color=[0, 192, 192], type='', swap=''),
        208: dict(name='midpoint_1_of_upper_inner_lip', id=208, color=[0, 192, 192], type='', swap='midpoint_2_of_upper_inner_lip'),
        209: dict(name='midpoint_2_of_upper_inner_lip', id=209, color=[0, 192, 192], type='', swap='midpoint_1_of_upper_inner_lip'),
        210: dict(name='midpoint_1_of_lower_inner_lip', id=210, color=[0, 192, 192], type='', swap='midpoint_2_of_lower_inner_lip'),
        211: dict(name='midpoint_2_of_lower_inner_lip', id=211, color=[0, 192, 192], type='', swap='midpoint_1_of_lower_inner_lip'),
        212: dict(name='midpoint_3_of_upper_inner_lip', id=212, color=[0, 192, 192], type='', swap='midpoint_6_of_upper_inner_lip'),
        213: dict(name='midpoint_4_of_upper_inner_lip', id=213, color=[0, 192, 192], type='', swap='midpoint_5_of_upper_inner_lip'),
        214: dict(name='midpoint_5_of_upper_inner_lip', id=214, color=[0, 192, 192], type='', swap='midpoint_4_of_upper_inner_lip'),
        215: dict(name='midpoint_6_of_upper_inner_lip', id=215, color=[0, 192, 192], type='', swap='midpoint_3_of_upper_inner_lip'),
        216: dict(name='midpoint_3_of_lower_inner_lip', id=216, color=[0, 192, 192], type='', swap='midpoint_6_of_lower_inner_lip'),
        217: dict(name='midpoint_4_of_lower_inner_lip', id=217, color=[0, 192, 192], type='', swap='midpoint_5_of_lower_inner_lip'),
        218: dict(name='midpoint_5_of_lower_inner_lip', id=218, color=[0, 192, 192], type='', swap='midpoint_4_of_lower_inner_lip'),
        219: dict(name='midpoint_6_of_lower_inner_lip', id=219, color=[0, 192, 192], type='', swap='midpoint_3_of_lower_inner_lip'),

        220: dict(name='teeth_1', id=220, color=[51, 153, 255], type='', swap=''),
        221: dict(name='teeth_2', id=221, color=[51, 153, 255], type='', swap=''),
        222: dict(name='teeth_3', id=222, color=[51, 153, 255], type='', swap=''),
        223: dict(name='teeth_4', id=223, color=[51, 153, 255], type='', swap=''),
        224: dict(name='teeth_5', id=224, color=[51, 153, 255], type='', swap=''),
        225: dict(name='teeth_6', id=225, color=[51, 153, 255], type='', swap=''),
        226: dict(name='teeth_7', id=226, color=[51, 153, 255], type='', swap=''),
        227: dict(name='teeth_8', id=227, color=[51, 153, 255], type='', swap=''),
        228: dict(name='teeth_9', id=228, color=[51, 153, 255], type='', swap=''),
        229: dict(name='teeth_10', id=229, color=[51, 153, 255], type='', swap=''),
        230: dict(name='teeth_11', id=230, color=[51, 153, 255], type='', swap=''),
        231: dict(name='teeth_12', id=231, color=[51, 153, 255], type='', swap=''),
        232: dict(name='teeth_13', id=232, color=[51, 153, 255], type='', swap=''),
        233: dict(name='teeth_14', id=233, color=[51, 153, 255], type='', swap=''),
        234: dict(name='teeth_15', id=234, color=[51, 153, 255], type='', swap=''),
        235: dict(name='teeth_16', id=235, color=[51, 153, 255], type='', swap=''),
        236: dict(name='teeth_17', id=236, color=[51, 153, 255], type='', swap=''),
        237: dict(name='teeth_18', id=237, color=[51, 153, 255], type='', swap=''),
        238: dict(name='teeth_19', id=238, color=[51, 153, 255], type='', swap=''),
        239: dict(name='teeth_20', id=239, color=[51, 153, 255], type='', swap=''),
        240: dict(name='teeth_21', id=240, color=[51, 153, 255], type='', swap=''),
        241: dict(name='teeth_22', id=241, color=[51, 153, 255], type='', swap=''),
        242: dict(name='teeth_23', id=242, color=[51, 153, 255], type='', swap=''),
        243: dict(name='teeth_24', id=243, color=[51, 153, 255], type='', swap=''),
        244: dict(name='teeth_25', id=244, color=[51, 153, 255], type='', swap=''),
        245: dict(name='teeth_26', id=245, color=[51, 153, 255], type='', swap=''),
        246: dict(name='teeth_27', id=246, color=[51, 153, 255], type='', swap=''),
        247: dict(name='teeth_28', id=247, color=[51, 153, 255], type='', swap=''),
        248: dict(name='teeth_29', id=248, color=[51, 153, 255], type='', swap=''),
        249: dict(name='teeth_30', id=249, color=[51, 153, 255], type='', swap=''),
        250: dict(name='teeth_31', id=250, color=[51, 153, 255], type='', swap=''),
        251: dict(name='teeth_32', id=251, color=[51, 153, 255], type='', swap=''),
        252: dict(name='teeth_33', id=252, color=[51, 153, 255], type='', swap=''),
        253: dict(name='teeth_34', id=253, color=[51, 153, 255], type='', swap=''),
        254: dict(name='teeth_35', id=254, color=[51, 153, 255], type='', swap=''),
        255: dict(name='teeth_36', id=255, color=[51, 153, 255], type='', swap=''),

        256: dict(name='l_top_end_of_inferior_crus', id=256, color=[200, 200, 0], type='', swap='r_top_end_of_inferior_crus'),
        257: dict(name='l_top_end_of_superior_crus', id=257, color=[200, 200, 0], type='', swap='r_top_end_of_superior_crus'),
        258: dict(name='l_start_of_antihelix', id=258, color=[200, 200, 0], type='', swap='r_start_of_antihelix'),
        259: dict(name='l_end_of_antihelix', id=259, color=[200, 200, 0], type='', swap='r_end_of_antihelix'),
        260: dict(name='l_midpoint_1_of_antihelix', id=260, color=[200, 200, 0], type='', swap='r_midpoint_1_of_antihelix'),
        261: dict(name='l_midpoint_1_of_inferior_crus', id=261, color=[200, 200, 0], type='', swap='r_midpoint_1_of_inferior_crus'),
        262: dict(name='l_midpoint_2_of_antihelix', id=262, color=[200, 200, 0], type='', swap='r_midpoint_2_of_antihelix'),
        263: dict(name='l_midpoint_3_of_antihelix', id=263, color=[200, 200, 0], type='', swap='r_midpoint_3_of_antihelix'),

        264: dict(name='l_point_1_of_inner_helix', id=264, color=[200, 200, 0], type='', swap='r_point_1_of_inner_helix'),
        265: dict(name='l_point_2_of_inner_helix', id=265, color=[200, 200, 0], type='', swap='r_point_8_of_inner_helix'),
        266: dict(name='l_point_3_of_inner_helix', id=266, color=[200, 200, 0], type='', swap='r_point_3_of_inner_helix'),
        267: dict(name='l_point_4_of_inner_helix', id=267, color=[200, 200, 0], type='', swap='r_point_4_of_inner_helix'),
        268: dict(name='l_point_5_of_inner_helix', id=268, color=[200, 200, 0], type='', swap='r_point_5_of_inner_helix'),
        269: dict(name='l_point_6_of_inner_helix', id=269, color=[200, 200, 0], type='', swap='r_point_6_of_inner_helix'),
        270: dict(name='l_point_7_of_inner_helix', id=270, color=[200, 200, 0], type='', swap='r_point_7_of_inner_helix'),
        271: dict(name='l_highest_point_of_antitragus', id=271, color=[200, 200, 0], type='', swap='r_highest_point_of_antitragus'),
        272: dict(name='l_bottom_point_of_tragus', id=272, color=[200, 200, 0], type='', swap='r_bottom_point_of_tragus'),
        273: dict(name='l_protruding_point_of_tragus', id=273, color=[200, 200, 0], type='', swap='r_protruding_point_of_tragus'),
        274: dict(name='l_top_point_of_tragus', id=274, color=[200, 200, 0], type='', swap='r_top_point_of_tragus'),
        275: dict(name='l_start_point_of_crus_of_helix', id=275, color=[200, 200, 0], type='', swap='r_start_point_of_crus_of_helix'),
        276: dict(name='l_deepest_point_of_concha', id=276, color=[200, 200, 0], type='', swap='r_deepest_point_of_concha'),
        277: dict(name='l_tip_of_ear_lobe', id=277, color=[200, 200, 0], type='', swap='r_tip_of_ear_lobe'),
        278: dict(name='l_midpoint_between_22_15', id=278, color=[200, 200, 0], type='', swap='r_midpoint_between_22_15'),
        279: dict(name='l_bottom_connecting_point_of_ear_lobe', id=279, color=[200, 200, 0], type='', swap='r_bottom_connecting_point_of_ear_lobe'),
        280: dict(name='l_top_connecting_point_of_helix', id=280, color=[200, 200, 0], type='', swap='r_top_connecting_point_of_helix'),
        281: dict(name='l_point_8_of_inner_helix', id=281, color=[200, 200, 0], type='', swap='r_point_2_of_inner_helix'),

        282: dict(name='r_top_end_of_inferior_crus', id=282, color=[0, 200, 200], type='', swap='l_top_end_of_inferior_crus'),
        283: dict(name='r_top_end_of_superior_crus', id=283, color=[0, 200, 200], type='', swap='l_top_end_of_superior_crus'),
        284: dict(name='r_start_of_antihelix', id=284, color=[0, 200, 200], type='', swap='l_start_of_antihelix'),
        285: dict(name='r_end_of_antihelix', id=285, color=[0, 200, 200], type='', swap='l_end_of_antihelix'),
        286: dict(name='r_midpoint_1_of_antihelix', id=286, color=[0, 200, 200], type='', swap='l_midpoint_1_of_antihelix'),
        287: dict(name='r_midpoint_1_of_inferior_crus', id=287, color=[0, 200, 200], type='', swap='l_midpoint_1_of_inferior_crus'),
        288: dict(name='r_midpoint_2_of_antihelix', id=288, color=[0, 200, 200], type='', swap='l_midpoint_2_of_antihelix'),
        289: dict(name='r_midpoint_3_of_antihelix', id=289, color=[0, 200, 200], type='', swap='l_midpoint_3_of_antihelix'),
        290: dict(name='r_point_1_of_inner_helix', id=290, color=[0, 200, 200], type='', swap='l_point_1_of_inner_helix'),
        291: dict(name='r_point_8_of_inner_helix', id=291, color=[0, 200, 200], type='', swap='l_point_2_of_inner_helix'),
        292: dict(name='r_point_3_of_inner_helix', id=292, color=[0, 200, 200], type='', swap='l_point_3_of_inner_helix'),
        293: dict(name='r_point_4_of_inner_helix', id=293, color=[0, 200, 200], type='', swap='l_point_4_of_inner_helix'),
        294: dict(name='r_point_5_of_inner_helix', id=294, color=[0, 200, 200], type='', swap='l_point_5_of_inner_helix'),
        295: dict(name='r_point_6_of_inner_helix', id=295, color=[0, 200, 200], type='', swap='l_point_6_of_inner_helix'),
        296: dict(name='r_point_7_of_inner_helix', id=296, color=[0, 200, 200], type='', swap='l_point_7_of_inner_helix'),

        297: dict(name='r_highest_point_of_antitragus', id=297, color=[0, 200, 200], type='', swap='l_highest_point_of_antitragus'),
        298: dict(name='r_bottom_point_of_tragus', id=298, color=[0, 200, 200], type='', swap='l_bottom_point_of_tragus'),
        299: dict(name='r_protruding_point_of_tragus', id=299, color=[0, 200, 200], type='', swap='l_protruding_point_of_tragus'),
        300: dict(name='r_top_point_of_tragus', id=300, color=[0, 200, 200], type='', swap='l_top_point_of_tragus'),
        301: dict(name='r_start_point_of_crus_of_helix', id=301, color=[0, 200, 200], type='', swap='l_start_point_of_crus_of_helix'),
        302: dict(name='r_deepest_point_of_concha', id=302, color=[0, 200, 200], type='', swap='l_deepest_point_of_concha'),
        303: dict(name='r_tip_of_ear_lobe', id=303, color=[0, 200, 200], type='', swap='l_tip_of_ear_lobe'),
        304: dict(name='r_midpoint_between_22_15', id=304, color=[0, 200, 200], type='', swap='l_midpoint_between_22_15'),
        305: dict(name='r_bottom_connecting_point_of_ear_lobe', id=305, color=[0, 200, 200], type='', swap='l_bottom_connecting_point_of_ear_lobe'),
        306: dict(name='r_top_connecting_point_of_helix', id=306, color=[0, 200, 200], type='', swap='l_top_connecting_point_of_helix'),
        307: dict(name='r_point_2_of_inner_helix', id=307, color=[0, 200, 200], type='', swap='l_point_8_of_inner_helix'),

        308: dict(name='l_center_of_iris', id=308, color=[128, 192, 64], type='', swap='r_center_of_iris'),
        309: dict(name='l_border_of_iris_3', id=309, color=[128, 192, 64], type='', swap='r_border_of_iris_9'),
        310: dict(name='l_border_of_iris_midpoint_1', id=310, color=[128, 192, 64], type='', swap='r_border_of_iris_midpoint_4'),
        311: dict(name='l_border_of_iris_12', id=311, color=[128, 192, 64], type='', swap='r_border_of_iris_12'),
        312: dict(name='l_border_of_iris_midpoint_4', id=312, color=[128, 192, 64], type='', swap='r_border_of_iris_midpoint_1'),
        313: dict(name='l_border_of_iris_9', id=313, color=[128, 192, 64], type='', swap='r_border_of_iris_3'),
        314: dict(name='l_border_of_iris_midpoint_3', id=314, color=[128, 192, 64], type='', swap='r_border_of_iris_midpoint_2'),
        315: dict(name='l_border_of_iris_6', id=315, color=[128, 192, 64], type='', swap='r_border_of_iris_6'),
        316: dict(name='l_border_of_iris_midpoint_2', id=316, color=[128, 192, 64], type='', swap='r_border_of_iris_midpoint_3'),
        317: dict(name='r_center_of_iris', id=317, color=[192, 32, 64], type='', swap='l_center_of_iris'),
        318: dict(name='r_border_of_iris_3', id=318, color=[192, 32, 64], type='', swap='l_border_of_iris_9'),
        319: dict(name='r_border_of_iris_midpoint_1', id=319, color=[192, 32, 64], type='', swap='l_border_of_iris_midpoint_4'),
        320: dict(name='r_border_of_iris_12', id=320, color=[192, 32, 64], type='', swap='l_border_of_iris_12'),
        321: dict(name='r_border_of_iris_midpoint_4', id=321, color=[192, 32, 64], type='', swap='l_border_of_iris_midpoint_1'),
        322: dict(name='r_border_of_iris_9', id=322, color=[192, 32, 64], type='', swap='l_border_of_iris_3'),
        323: dict(name='r_border_of_iris_midpoint_3', id=323, color=[192, 32, 64], type='', swap='l_border_of_iris_midpoint_2'),

        324: dict(name='r_border_of_iris_6', id=324, color=[192, 32, 64], type='', swap='l_border_of_iris_6'),
        325: dict(name='r_border_of_iris_midpoint_2', id=325, color=[192, 32, 64], type='', swap='l_border_of_iris_midpoint_3'),
        326: dict(name='l_center_of_pupil', id=326, color=[192, 128, 64], type='', swap='r_center_of_pupil'),
        327: dict(name='l_border_of_pupil_3', id=327, color=[192, 128, 64], type='', swap='r_border_of_pupil_9'),
        328: dict(name='l_border_of_pupil_midpoint_1', id=328, color=[192, 128, 64], type='', swap='r_border_of_pupil_midpoint_4'),
        329: dict(name='l_border_of_pupil_12', id=329, color=[192, 128, 64], type='', swap='r_border_of_pupil_12'),
        330: dict(name='l_border_of_pupil_midpoint_4', id=330, color=[192, 128, 64], type='', swap='r_border_of_pupil_midpoint_1'),
        331: dict(name='l_border_of_pupil_9', id=331, color=[192, 128, 64], type='', swap='r_border_of_pupil_3'),
        332: dict(name='l_border_of_pupil_midpoint_3', id=332, color=[192, 128, 64], type='', swap='r_border_of_pupil_midpoint_2'),
        333: dict(name='l_border_of_pupil_6', id=333, color=[192, 128, 64], type='', swap='r_border_of_pupil_6'),
        334: dict(name='l_border_of_pupil_midpoint_2', id=334, color=[192, 128, 64], type='', swap='r_border_of_pupil_midpoint_3'),

        335: dict(name='r_center_of_pupil', id=335, color=[32, 192, 192], type='', swap='l_center_of_pupil'),
        336: dict(name='r_border_of_pupil_3', id=336, color=[32, 192, 192], type='', swap='l_border_of_pupil_9'),
        337: dict(name='r_border_of_pupil_midpoint_1', id=337, color=[32, 192, 192], type='', swap='l_border_of_pupil_midpoint_4'),
        338: dict(name='r_border_of_pupil_12', id=338, color=[32, 192, 192], type='', swap='l_border_of_pupil_12'),
        339: dict(name='r_border_of_pupil_midpoint_4', id=339, color=[32, 192, 192], type='', swap='l_border_of_pupil_midpoint_1'),
        340: dict(name='r_border_of_pupil_9', id=340, color=[32, 192, 192], type='', swap='l_border_of_pupil_3'),
        341: dict(name='r_border_of_pupil_midpoint_3', id=341, color=[32, 192, 192], type='', swap='l_border_of_pupil_midpoint_2'),
        342: dict(name='r_border_of_pupil_6', id=342, color=[32, 192, 192], type='', swap='l_border_of_pupil_6'),
        343: dict(name='r_border_of_pupil_midpoint_2', id=343, color=[32, 192, 192], type='', swap='l_border_of_pupil_midpoint_3'),
    },
    remove_teeth=True, ## 36 of them, 344 - 36 = 308
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        18:
        dict(
            link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255]),
        19:
        dict(link=('left_ankle', 'left_big_toe'), id=19, color=[0, 255, 0]),
        20:
        dict(link=('left_ankle', 'left_small_toe'), id=20, color=[0, 255, 0]),
        21:
        dict(link=('left_ankle', 'left_heel'), id=21, color=[0, 255, 0]),
        22:
        dict(
            link=('right_ankle', 'right_big_toe'), id=22, color=[255, 128, 0]),
        23:
        dict(
            link=('right_ankle', 'right_small_toe'),
            id=23,
            color=[255, 128, 0]),
        24:
        dict(link=('right_ankle', 'right_heel'), id=24, color=[255, 128, 0]),
        25:
        dict(
            link=('left_wrist', 'left_thumb_third_joint'), id=25, color=[255, 128,
                                                                  0]),
        26:
        dict(link=('left_thumb_third_joint', 'left_thumb2'), id=26, color=[255, 128, 0]),
        27:
        dict(link=('left_thumb2', 'left_thumb3'), id=27, color=[255, 128, 0]),
        28:
        dict(link=('left_thumb3', 'left_thumb4'), id=28, color=[255, 128, 0]),
        29:
        dict(
            link=('left_wrist', 'left_forefinger_third_joint'),
            id=29,
            color=[255, 153, 255]),
        30:
        dict(
            link=('left_forefinger_third_joint', 'left_forefinger2'),
            id=30,
            color=[255, 153, 255]),
        31:
        dict(
            link=('left_forefinger2', 'left_forefinger3'),
            id=31,
            color=[255, 153, 255]),
        32:
        dict(
            link=('left_forefinger3', 'left_forefinger4'),
            id=32,
            color=[255, 153, 255]),
        33:
        dict(
            link=('left_wrist', 'left_middle_finger_third_joint'),
            id=33,
            color=[102, 178, 255]),
        34:
        dict(
            link=('left_middle_finger_third_joint', 'left_middle_finger2'),
            id=34,
            color=[102, 178, 255]),
        35:
        dict(
            link=('left_middle_finger2', 'left_middle_finger3'),
            id=35,
            color=[102, 178, 255]),
        36:
        dict(
            link=('left_middle_finger3', 'left_middle_finger4'),
            id=36,
            color=[102, 178, 255]),
        37:
        dict(
            link=('left_wrist', 'left_ring_finger_third_joint'),
            id=37,
            color=[255, 51, 51]),
        38:
        dict(
            link=('left_ring_finger_third_joint', 'left_ring_finger2'),
            id=38,
            color=[255, 51, 51]),
        39:
        dict(
            link=('left_ring_finger2', 'left_ring_finger3'),
            id=39,
            color=[255, 51, 51]),
        40:
        dict(
            link=('left_ring_finger3', 'left_ring_finger4'),
            id=40,
            color=[255, 51, 51]),
        41:
        dict(
            link=('left_wrist', 'left_pinky_finger_third_joint'),
            id=41,
            color=[0, 255, 0]),
        42:
        dict(
            link=('left_pinky_finger_third_joint', 'left_pinky_finger2'),
            id=42,
            color=[0, 255, 0]),
        43:
        dict(
            link=('left_pinky_finger2', 'left_pinky_finger3'),
            id=43,
            color=[0, 255, 0]),
        44:
        dict(
            link=('left_pinky_finger3', 'left_pinky_finger4'),
            id=44,
            color=[0, 255, 0]),
        45:
        dict(
            link=('right_wrist', 'right_thumb_third_joint'),
            id=45,
            color=[255, 128, 0]),
        46:
        dict(
            link=('right_thumb_third_joint', 'right_thumb2'), id=46, color=[255, 128, 0]),
        47:
        dict(
            link=('right_thumb2', 'right_thumb3'), id=47, color=[255, 128, 0]),
        48:
        dict(
            link=('right_thumb3', 'right_thumb4'), id=48, color=[255, 128, 0]),
        49:
        dict(
            link=('right_wrist', 'right_forefinger_third_joint'),
            id=49,
            color=[255, 153, 255]),
        50:
        dict(
            link=('right_forefinger_third_joint', 'right_forefinger2'),
            id=50,
            color=[255, 153, 255]),
        51:
        dict(
            link=('right_forefinger2', 'right_forefinger3'),
            id=51,
            color=[255, 153, 255]),
        52:
        dict(
            link=('right_forefinger3', 'right_forefinger4'),
            id=52,
            color=[255, 153, 255]),
        53:
        dict(
            link=('right_wrist', 'right_middle_finger_third_joint'),
            id=53,
            color=[102, 178, 255]),
        54:
        dict(
            link=('right_middle_finger_third_joint', 'right_middle_finger2'),
            id=54,
            color=[102, 178, 255]),
        55:
        dict(
            link=('right_middle_finger2', 'right_middle_finger3'),
            id=55,
            color=[102, 178, 255]),
        56:
        dict(
            link=('right_middle_finger3', 'right_middle_finger4'),
            id=56,
            color=[102, 178, 255]),
        57:
        dict(
            link=('right_wrist', 'right_ring_finger_third_joint'),
            id=57,
            color=[255, 51, 51]),
        58:
        dict(
            link=('right_ring_finger_third_joint', 'right_ring_finger2'),
            id=58,
            color=[255, 51, 51]),
        59:
        dict(
            link=('right_ring_finger2', 'right_ring_finger3'),
            id=59,
            color=[255, 51, 51]),
        60:
        dict(
            link=('right_ring_finger3', 'right_ring_finger4'),
            id=60,
            color=[255, 51, 51]),
        61:
        dict(
            link=('right_wrist', 'right_pinky_finger_third_joint'),
            id=61,
            color=[0, 255, 0]),
        62:
        dict(
            link=('right_pinky_finger_third_joint', 'right_pinky_finger2'),
            id=62,
            color=[0, 255, 0]),
        63:
        dict(
            link=('right_pinky_finger2', 'right_pinky_finger3'),
            id=63,
            color=[0, 255, 0]),
        64:
        dict(
            link=('right_pinky_finger3', 'right_pinky_finger4'),
            id=64,
            color=[0, 255, 0])
    },
    joint_weights=[1.] * 344,
    body_keypoint_names=[
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
    ],
    foot_keypoint_names=[
        'left_big_toe',
        'left_small_toe',
        'left_heel',
        'right_big_toe',
        'right_small_toe',
        'right_heel'
    ],
    left_hand_keypoint_names=[
        'left_thumb4',
        'left_thumb3',
        'left_thumb2',
        'left_thumb_third_joint',
        'left_forefinger4',
        'left_forefinger3',
        'left_forefinger2',
        'left_forefinger_third_joint',
        'left_middle_finger4',
        'left_middle_finger3',
        'left_middle_finger2',
        'left_middle_finger_third_joint',
        'left_ring_finger4',
        'left_ring_finger3',
        'left_ring_finger2',
        'left_ring_finger_third_joint',
        'left_pinky_finger4',
        'left_pinky_finger3',
        'left_pinky_finger2',
        'left_pinky_finger_third_joint'
    ],
    right_hand_keypoint_names=[
        'right_thumb4',
        'right_thumb3',
        'right_thumb2',
        'right_thumb_third_joint',
        'right_forefinger4',
        'right_forefinger3',
        'right_forefinger2',
        'right_forefinger_third_joint',
        'right_middle_finger4',
        'right_middle_finger3',
        'right_middle_finger2',
        'right_middle_finger_third_joint',
        'right_ring_finger4',
        'right_ring_finger3',
        'right_ring_finger2',
        'right_ring_finger_third_joint',
        'right_pinky_finger4',
        'right_pinky_finger3',
        'right_pinky_finger2',
        'right_pinky_finger_third_joint'
    ],
    ## 7 of them
    extra_keypoint_names=[
        'neck',
        'left_olecranon',
        'right_olecranon',
        'left_cubital_fossa',
        'right_cubital_fossa',
        'left_acromion',
        'right_acromion',
    ],
    face_keypoint_names=[
        'center_of_glabella',
        'center_of_nose_root',
        'tip_of_nose_bridge',
        'midpoint_1_of_nose_bridge',
        'midpoint_2_of_nose_bridge',
        'midpoint_3_of_nose_bridge',
        'center_of_labiomental_groove',
        'tip_of_chin',
        'upper_startpoint_of_r_eyebrow',
        'lower_startpoint_of_r_eyebrow',
        'end_of_r_eyebrow',
        'upper_midpoint_1_of_r_eyebrow',
        'lower_midpoint_1_of_r_eyebrow',
        'upper_midpoint_2_of_r_eyebrow',
        'upper_midpoint_3_of_r_eyebrow',
        'lower_midpoint_2_of_r_eyebrow',
        'lower_midpoint_3_of_r_eyebrow',
        'upper_startpoint_of_l_eyebrow',
        'lower_startpoint_of_l_eyebrow',
        'end_of_l_eyebrow',
        'upper_midpoint_1_of_l_eyebrow',
        'lower_midpoint_1_of_l_eyebrow',
        'upper_midpoint_2_of_l_eyebrow',
        'upper_midpoint_3_of_l_eyebrow',
        'lower_midpoint_2_of_l_eyebrow',
        'lower_midpoint_3_of_l_eyebrow',
        'l_inner_end_of_upper_lash_line',
        'l_outer_end_of_upper_lash_line',
        'l_centerpoint_of_upper_lash_line',
        'l_midpoint_2_of_upper_lash_line',
        'l_midpoint_1_of_upper_lash_line',
        'l_midpoint_6_of_upper_lash_line',
        'l_midpoint_5_of_upper_lash_line',
        'l_midpoint_4_of_upper_lash_line',
        'l_midpoint_3_of_upper_lash_line',
        'l_outer_end_of_upper_eyelid_line',
        'l_midpoint_6_of_upper_eyelid_line',
        'l_midpoint_2_of_upper_eyelid_line',
        'l_midpoint_5_of_upper_eyelid_line',
        'l_centerpoint_of_upper_eyelid_line',
        'l_midpoint_4_of_upper_eyelid_line',
        'l_midpoint_1_of_upper_eyelid_line',
        'l_midpoint_3_of_upper_eyelid_line',
        'l_midpoint_6_of_upper_crease_line',
        'l_midpoint_2_of_upper_crease_line',
        'l_midpoint_5_of_upper_crease_line',
        'l_centerpoint_of_upper_crease_line',
        'l_midpoint_4_of_upper_crease_line',
        'l_midpoint_1_of_upper_crease_line',
        'l_midpoint_3_of_upper_crease_line',
        'r_inner_end_of_upper_lash_line',
        'r_outer_end_of_upper_lash_line',
        'r_centerpoint_of_upper_lash_line',
        'r_midpoint_1_of_upper_lash_line',
        'r_midpoint_2_of_upper_lash_line',
        'r_midpoint_3_of_upper_lash_line',
        'r_midpoint_4_of_upper_lash_line',
        'r_midpoint_5_of_upper_lash_line',
        'r_midpoint_6_of_upper_lash_line',
        'r_outer_end_of_upper_eyelid_line',
        'r_midpoint_3_of_upper_eyelid_line',
        'r_midpoint_1_of_upper_eyelid_line',
        'r_midpoint_4_of_upper_eyelid_line',
        'r_centerpoint_of_upper_eyelid_line',
        'r_midpoint_5_of_upper_eyelid_line',
        'r_midpoint_2_of_upper_eyelid_line',
        'r_midpoint_6_of_upper_eyelid_line',
        'r_midpoint_3_of_upper_crease_line',
        'r_midpoint_1_of_upper_crease_line',
        'r_midpoint_4_of_upper_crease_line',
        'r_centerpoint_of_upper_crease_line',
        'r_midpoint_5_of_upper_crease_line',
        'r_midpoint_2_of_upper_crease_line',
        'r_midpoint_6_of_upper_crease_line',
        'l_inner_end_of_lower_lash_line',
        'l_outer_end_of_lower_lash_line',
        'l_centerpoint_of_lower_lash_line',
        'l_midpoint_2_of_lower_lash_line',
        'l_midpoint_1_of_lower_lash_line',
        'l_midpoint_6_of_lower_lash_line',
        'l_midpoint_5_of_lower_lash_line',
        'l_midpoint_4_of_lower_lash_line',
        'l_midpoint_3_of_lower_lash_line',
        'l_outer_end_of_lower_eyelid_line',
        'l_midpoint_6_of_lower_eyelid_line',
        'l_midpoint_2_of_lower_eyelid_line',
        'l_midpoint_5_of_lower_eyelid_line',
        'l_centerpoint_of_lower_eyelid_line',
        'l_midpoint_4_of_lower_eyelid_line',
        'l_midpoint_1_of_lower_eyelid_line',
        'l_midpoint_3_of_lower_eyelid_line',
        'r_inner_end_of_lower_lash_line',
        'r_outer_end_of_lower_lash_line',
        'r_centerpoint_of_lower_lash_line',
        'r_midpoint_1_of_lower_lash_line',
        'r_midpoint_2_of_lower_lash_line',
        'r_midpoint_3_of_lower_lash_line',
        'r_midpoint_4_of_lower_lash_line',
        'r_midpoint_5_of_lower_lash_line',
        'r_midpoint_6_of_lower_lash_line',
        'r_outer_end_of_lower_eyelid_line',
        'r_midpoint_3_of_lower_eyelid_line',
        'r_midpoint_1_of_lower_eyelid_line',
        'r_midpoint_4_of_lower_eyelid_line',
        'r_centerpoint_of_lower_eyelid_line',
        'r_midpoint_5_of_lower_eyelid_line',
        'r_midpoint_2_of_lower_eyelid_line',
        'r_midpoint_6_of_lower_eyelid_line',
        'tip_of_nose',
        'bottom_center_of_nose',
        'r_outer_corner_of_nose',
        'l_outer_corner_of_nose',
        'inner_corner_of_r_nostril',
        'outer_corner_of_r_nostril',
        'upper_corner_of_r_nostril',
        'inner_corner_of_l_nostril',
        'outer_corner_of_l_nostril',
        'upper_corner_of_l_nostril',
        'r_outer_corner_of_mouth',
        'l_outer_corner_of_mouth',
        'center_of_cupid_bow',
        'center_of_lower_outer_lip',
        'midpoint_1_of_upper_outer_lip',
        'midpoint_2_of_upper_outer_lip',
        'midpoint_1_of_lower_outer_lip',
        'midpoint_2_of_lower_outer_lip',
        'midpoint_3_of_upper_outer_lip',
        'midpoint_4_of_upper_outer_lip',
        'midpoint_5_of_upper_outer_lip',
        'midpoint_6_of_upper_outer_lip',
        'midpoint_3_of_lower_outer_lip',
        'midpoint_4_of_lower_outer_lip',
        'midpoint_5_of_lower_outer_lip',
        'midpoint_6_of_lower_outer_lip',
        'r_inner_corner_of_mouth',
        'l_inner_corner_of_mouth',
        'center_of_upper_inner_lip',
        'center_of_lower_inner_lip',
        'midpoint_1_of_upper_inner_lip',
        'midpoint_2_of_upper_inner_lip',
        'midpoint_1_of_lower_inner_lip',
        'midpoint_2_of_lower_inner_lip',
        'midpoint_3_of_upper_inner_lip',
        'midpoint_4_of_upper_inner_lip',
        'midpoint_5_of_upper_inner_lip',
        'midpoint_6_of_upper_inner_lip',
        'midpoint_3_of_lower_inner_lip',
        'midpoint_4_of_lower_inner_lip',
        'midpoint_5_of_lower_inner_lip',
        'midpoint_6_of_lower_inner_lip',
        'l_top_end_of_inferior_crus',
        'l_top_end_of_superior_crus',
        'l_start_of_antihelix',
        'l_end_of_antihelix',
        'l_midpoint_1_of_antihelix',
        'l_midpoint_1_of_inferior_crus',
        'l_midpoint_2_of_antihelix',
        'l_midpoint_3_of_antihelix',
        'l_point_1_of_inner_helix',
        'l_point_2_of_inner_helix',
        'l_point_3_of_inner_helix',
        'l_point_4_of_inner_helix',
        'l_point_5_of_inner_helix',
        'l_point_6_of_inner_helix',
        'l_point_7_of_inner_helix',
        'l_highest_point_of_antitragus',
        'l_bottom_point_of_tragus',
        'l_protruding_point_of_tragus',
        'l_top_point_of_tragus',
        'l_start_point_of_crus_of_helix',
        'l_deepest_point_of_concha',
        'l_tip_of_ear_lobe',
        'l_midpoint_between_22_15',
        'l_bottom_connecting_point_of_ear_lobe',
        'l_top_connecting_point_of_helix',
        'l_point_8_of_inner_helix',
        'r_top_end_of_inferior_crus',
        'r_top_end_of_superior_crus',
        'r_start_of_antihelix',
        'r_end_of_antihelix',
        'r_midpoint_1_of_antihelix',
        'r_midpoint_1_of_inferior_crus',
        'r_midpoint_2_of_antihelix',
        'r_midpoint_3_of_antihelix',
        'r_point_1_of_inner_helix',
        'r_point_8_of_inner_helix',
        'r_point_3_of_inner_helix',
        'r_point_4_of_inner_helix',
        'r_point_5_of_inner_helix',
        'r_point_6_of_inner_helix',
        'r_point_7_of_inner_helix',
        'r_highest_point_of_antitragus',
        'r_bottom_point_of_tragus',
        'r_protruding_point_of_tragus',
        'r_top_point_of_tragus',
        'r_start_point_of_crus_of_helix',
        'r_deepest_point_of_concha',
        'r_tip_of_ear_lobe',
        'r_midpoint_between_22_15',
        'r_bottom_connecting_point_of_ear_lobe',
        'r_top_connecting_point_of_helix',
        'r_point_2_of_inner_helix',
        'l_center_of_iris',
        'l_border_of_iris_3',
        'l_border_of_iris_midpoint_1',
        'l_border_of_iris_12',
        'l_border_of_iris_midpoint_4',
        'l_border_of_iris_9',
        'l_border_of_iris_midpoint_3',
        'l_border_of_iris_6',
        'l_border_of_iris_midpoint_2',
        'r_center_of_iris',
        'r_border_of_iris_3',
        'r_border_of_iris_midpoint_1',
        'r_border_of_iris_12',
        'r_border_of_iris_midpoint_4',
        'r_border_of_iris_9',
        'r_border_of_iris_midpoint_3',
        'r_border_of_iris_6',
        'r_border_of_iris_midpoint_2',
        'l_center_of_pupil',
        'l_border_of_pupil_3',
        'l_border_of_pupil_midpoint_1',
        'l_border_of_pupil_12',
        'l_border_of_pupil_midpoint_4',
        'l_border_of_pupil_9',
        'l_border_of_pupil_midpoint_3',
        'l_border_of_pupil_6',
        'l_border_of_pupil_midpoint_2',
        'r_center_of_pupil',
        'r_border_of_pupil_3',
        'r_border_of_pupil_midpoint_1',
        'r_border_of_pupil_12',
        'r_border_of_pupil_midpoint_4',
        'r_border_of_pupil_9',
        'r_border_of_pupil_midpoint_3',
        'r_border_of_pupil_6',
        'r_border_of_pupil_midpoint_2',
    ]
)

##------------------------------------------------------------------------------------------------------------------
### remove teeth keypoints
if dataset_info['remove_teeth'] == True:

    ## get teeth ids
    teeth_keypoint_ids = [keypoint_id for keypoint_id, info in dataset_info['keypoint_info'].items() if info['name'].startswith('teeth_')]
    min_teeth_keypoint_id = min(teeth_keypoint_ids)
    max_teeth_keypoint_id = max(teeth_keypoint_ids)

    dataset_info['teeth_keypoint_ids'] = teeth_keypoint_ids

    ## remove teeth keypoints from keypoint_info
    keypoint_info_ = dataset_info['keypoint_info']

    keypoint_info = {}
    for keypoint_id, info in keypoint_info_.items():
        if keypoint_id in teeth_keypoint_ids:
            continue

        if keypoint_id < min_teeth_keypoint_id:
            keypoint_info[keypoint_id] = info

        if keypoint_id > max_teeth_keypoint_id:

            keypoint_id = keypoint_id - len(teeth_keypoint_ids)
            info['id'] = keypoint_id
            keypoint_info[keypoint_id] = info

    dataset_info['keypoint_info'] = keypoint_info

    ## update joint_weights
    dataset_info['joint_weights'] = (len(keypoint_info_) - len(teeth_keypoint_ids))*[1.]

##------------------------------------------------------------------------------------------------------------------
## reconfigure in the order of coco_whole_body
coco_wholebody_keypoint_info = {keypoint_info['name']: keypoint_info for (keypoint_index, keypoint_info) in coco_wholebody_info['keypoint_info'].items()}
coco_wholebody_to_goliath_mapping = {} ## coco_wholebody_index to goliath_index
coco_wholebody_to_goliath_keypoint_info = {}

## find out common keypoints between goliath and coco_whole_body
for (keypoint_index, keypoint_info) in dataset_info['keypoint_info'].items():
    keypoint_name = keypoint_info['name']
    keypoint_index_ = keypoint_info['id']
    assert(keypoint_index == keypoint_index_)

    if keypoint_name in coco_wholebody_keypoint_info.keys():
        coco_wholebody_to_goliath_keypoint_info[keypoint_name] = coco_wholebody_keypoint_info[keypoint_name]
        coco_wholebody_to_goliath_mapping[coco_wholebody_keypoint_info[keypoint_name]['id']] = keypoint_info['id']

dataset_info['coco_wholebody_to_goliath_mapping'] = coco_wholebody_to_goliath_mapping ## store the cocowholebody indices
dataset_info['coco_wholebody_to_goliath_keypoint_info'] = coco_wholebody_to_goliath_keypoint_info

##------------------------------------------------------------------------------------------------------------------
coco_wholebody_sigmas = {}

## compute the coco_wholebody_sigmas
for keypoint_index, keypoint_info in coco_wholebody_info['keypoint_info'].items():
    coco_wholebody_sigmas[keypoint_info['name']] = coco_wholebody_info['sigmas'][keypoint_info['id']]

default_sigma = 0.010 ## for mostly face keypoints
dataset_info['sigmas'] = [default_sigma]*len(dataset_info['keypoint_info'])

## we copy sigmas from coco_wholebody. Rest are assigned as below:
custom_sigmas = {
    'left_thumb_third_joint': 0.022,
    'left_forefinger_third_joint': 0.026,
    'left_middle_finger_third_joint': 0.018,
    'left_ring_finger_third_joint': 0.017,
    'left_pinky_finger_third_joint': 0.02,
    'right_thumb_third_joint': 0.022,
    'right_forefinger_third_joint': 0.026,
    'right_middle_finger_third_joint': 0.018,
    'right_ring_finger_third_joint': 0.017,
    'right_pinky_finger_third_joint': 0.02,
    'neck': 0.079, ## same as shoulder
    'left_olecranon': 0.072, ## same as elbow
    'right_olecranon': 0.072, ## same as elbow
    'left_cubital_fossa': 0.072, ## same as elbow
    'right_cubital_fossa': 0.072, ## same as elbow
    'left_acromion': 0.079, ## same as shoulder
    'right_acromion': 0.079, ## same as shoulder
}

## copy custom sigmas
for keypoint_name, sigma in custom_sigmas.items():
    keypoint_id = -1

    ## search for keypoint id from keypoint name
    for keypoint_id_ in dataset_info['keypoint_info'].keys():
        if dataset_info['keypoint_info'][keypoint_id_]['name'] == keypoint_name:
            keypoint_id = keypoint_id_
            break

    if keypoint_id != -1:
        keypoint_info = dataset_info['keypoint_info'][keypoint_id]
        assert(keypoint_info['name'] == keypoint_name)
        assert(keypoint_info['id'] == keypoint_id)
        dataset_info['sigmas'][keypoint_info['id']] = sigma

## copy coco_wholebody sigmas
for keypoint_index, keypoint_info in dataset_info['keypoint_info'].items():
    if keypoint_info['name'] in coco_wholebody_sigmas.keys():
        dataset_info['sigmas'][keypoint_info['id']] = coco_wholebody_sigmas[keypoint_info['name']]
