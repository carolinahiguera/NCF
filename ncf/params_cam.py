import numpy as np

def get_cam_parameters(tj_id):
    if tj_id=="0_1_0":
        cameras = []
        cameras.append( dict(
            pos=(0.02726, -0.1367, -0.2574),
            focalPoint=(-7.924e-3, -3.187e-3, 8.751e-3),
            viewup=(0.9470, 0.3192, -0.03490),
            distance=0.2998,
            clippingRange=(0.1190, 0.5110),
        ) )

        cameras.append( dict(
            pos=(0.1621, -0.05860, -0.2319),
            focalPoint=(-7.920e-3, -3.202e-3, 8.721e-3),
            viewup=(0.7634, 0.4836, 0.4281),
            distance=0.2998,
            clippingRange=(0.09860, 0.5336),
        ) )

        cameras.append( dict(
            pos=(0.2333, 0.01155, -0.1687),
            focalPoint=(-7.920e-3, -3.202e-3, 8.721e-3),
            viewup=(0.4865, 0.5170, 0.7043),
            distance=0.2998,
            clippingRange=(0.1090, 0.5219),
        ) )

        cameras.append( dict(
            pos=(0.2689, 0.08615, -0.06370),
            focalPoint=(-7.920e-3, -3.202e-3, 8.721e-3),
            viewup=(0.08933, 0.4453, 0.8909),
            distance=0.2998,
            clippingRange=(0.1116, 0.5236),
        ) )

        cameras.append( dict(
            pos=(0.2819, 0.04735, 0.06653),
            focalPoint=(-7.920e-3, -3.202e-3, 8.721e-3),
            viewup=(-0.2445, 0.3825, 0.8910),
            distance=0.2998,
            clippingRange=(0.1265, 0.5062),
        ) )

        cameras.append( dict(
            pos=(0.2500, -0.01208, 0.1614),
            focalPoint=(-7.899e-3, -3.198e-3, 8.725e-3),
            viewup=(-0.4638, 0.3699, 0.8050),
            distance=0.2998,
            clippingRange=(0.1175, 0.5192),
        ) )

        cameras.append( dict(
            pos=(0.1899, -0.08152, 0.2200),
            focalPoint=(-7.899e-3, -3.198e-3, 8.725e-3),
            viewup=(-0.5904, 0.3999, 0.7011),
            distance=0.2998,
            clippingRange=(0.09532, 0.5490),
        ) )

        cameras.append( dict(
            pos=(0.1258, -0.1681, 0.2204),
            focalPoint=(-7.899e-3, -3.198e-3, 8.725e-3),
            viewup=(-0.5979, 0.4039, 0.6924),
            distance=0.2998,
            clippingRange=(0.09107, 0.5537),
        ) )

        cameras.append( dict(
            pos=(0.09357, -0.2170, 0.1928),
            focalPoint=(-7.899e-3, -3.198e-3, 8.725e-3),
            viewup=(-0.5736, 0.3610, 0.7353),
            distance=0.2998,
            clippingRange=(0.09777, 0.5438),
        ) )

        t_switch_camera = np.zeros(49)
        sw = np.linspace(28, 44, len(cameras)-1, dtype=np.int8)
        for i in range(len(sw)-1):
            t_switch_camera[sw[i]:sw[i+1]] = i+1
        t_switch_camera[sw[-1]:] = len(cameras)-1
    
    elif tj_id=="1_1_4":
        cameras = []
        
        cameras.append( dict(
            pos=(0.1412, -0.1896, -0.4666),
            focalPoint=(-0.02450, 5.606e-3, -0.03119),
            viewup=(0.8389, -0.3005, 0.4539),
            distance=0.5051,
            clippingRange=(0.2465, 0.8319),
        ) )

        cameras.append( dict(
            pos=(0.2562, -0.2240, -0.3828),
            focalPoint=(-0.02450, 5.606e-3, -0.03119),
            viewup=(0.7088, -0.1787, 0.6825),
            distance=0.5051,
            clippingRange=(0.2432, 0.8360),
        ) )

        cameras.append( dict(
            pos=(0.3666, -0.2656, -0.2005),
            focalPoint=(-0.02450, 5.606e-3, -0.03119),
            viewup=(0.4214, 0.04212, 0.9059),
            distance=0.5051,
            clippingRange=(0.2798, 0.7900),
        ) )

        cameras.append( dict(
            pos=(0.4225, -0.2270, 3.646e-3),
            focalPoint=(-0.02450, 5.606e-3, -0.03119),
            viewup=(0.05995, 0.2595, 0.9639),
            distance=0.5051,
            clippingRange=(0.3252, 0.7328),
        ) )

        cameras.append( dict(
            pos=(0.3978, -0.1622, 0.1894),
            focalPoint=(-0.02450, 5.606e-3, -0.03119),
            viewup=(-0.2867, 0.4141, 0.8639),
            distance=0.5051,
            clippingRange=(0.2678, 0.8051),
        ) )

        # 15
        cameras.append( dict(
            pos=(0.3615, -0.1149, 0.2715),
            focalPoint=(-0.02448, 5.596e-3, -0.03118),
            viewup=(-0.4480, 0.4721, 0.7592),
            distance=0.5051,
            clippingRange=(0.2524, 0.8245),
        ) )

        # 16
        cameras.append( dict(
            pos=(0.4256, -0.1867, 0.09350),
            focalPoint=(-0.02448, 5.596e-3, -0.03118),
            viewup=(-0.1093, 0.3478, 0.9312),
            distance=0.5051,
            clippingRange=(0.2948, 0.7711),
        ) )

        # 17-18
        cameras.append( dict(
            pos=(0.3165520, -0.03610758, -0.4014767),
            focalPoint=(-0.02446238, 5.576052e-3, -0.03120974),
            viewup=(0.7092637, -0.2005688, 0.6758085),
            distance=0.5051000,
            clippingRange=(0.2671189, 0.7835852),
        ) )

        cameras.append( dict(
            pos=(0.2913821, -0.1174274, -0.4056939),
            focalPoint=(-0.02446242, 5.576057e-3, -0.03120970),
            viewup=(0.7149527, -0.2020126, 0.6693530),
            distance=0.5051000,
            clippingRange=(0.2421690, 0.8373998),
        ) )

        cameras.append( dict(
            pos=(0.2558193, -0.2007842, -0.3972474),
            focalPoint=(-0.02446242, 5.576057e-3, -0.03120970),
            viewup=(0.7183005, -0.2041478, 0.6651076),
            distance=0.5051000,
            clippingRange=(0.2414108, 0.8383590),
        ) )

        cameras.append( dict(
            pos=(0.1785018, -0.1717434, -0.4583974),
            focalPoint=(-0.02446242, 5.576057e-3, -0.03120970),
            viewup=(0.8159304, -0.2819946, 0.5047144),
            distance=0.5051000,
            clippingRange=(0.2426678, 0.8367670),
        ) )

        

        t_switch_camera = np.zeros(49)
        sw = np.linspace(0, 15, 6, dtype=np.int8)
        for i in range(len(sw)-1):
            t_switch_camera[sw[i]:sw[i+1]] = i
        t_switch_camera[15] = 5    
        t_switch_camera[16] = 6
        t_switch_camera[17] = 7
        t_switch_camera[18] = 7
        t_switch_camera[19:22] = 8
        t_switch_camera[22:25] = 9
        t_switch_camera[25:] = 10

    elif tj_id=="2_1_0":
        cameras = []
        
        # 0
        cameras.append( dict(
            pos=(-0.07740, -0.01475, -0.4600),
            focalPoint=(-8.459e-3, 5.653e-4, -5.102e-3),
            viewup=(-0.9748, 0.1720, 0.1419),
            distance=0.4604,
            clippingRange=(0.2827, 0.6851),
        ) )

        # 1
        cameras.append( dict(
            pos=(-0.2463, 0.01357, -0.3990),
            focalPoint=(-8.459e-3, 5.653e-4, -5.102e-3),
            viewup=(-0.8411, 0.1700, 0.5135),
            distance=0.4604,
            clippingRange=(0.2742, 0.6958),
        ) )

        # 2
        cameras.append( dict(
            pos=(-0.2872353, -4.774284e-5, 0.3612750),
            focalPoint=(-8.399294e-3, 5.590901e-4, -5.083153e-3),
            viewup=(0.7919977, 0.09586844, 0.6029501),
            distance=0.4604000,
            clippingRange=(0.2804209, 0.6876690),
        ) )

        # 3
        cameras.append( dict(
            pos=(-0.3566, 0.09230, 0.2817),
            focalPoint=(-8.459e-3, 5.653e-4, -5.102e-3),
            viewup=(0.6310, -0.02896, 0.7752),
            distance=0.4604,
            clippingRange=(0.2642, 0.7084),
        ) )

        # 4
        cameras.append( dict(
            pos=(-0.1946, 0.04767, 0.4133),
            focalPoint=(-8.459e-3, 5.653e-4, -5.102e-3),
            viewup=(0.9057, -0.09317, 0.4135),
            distance=0.4604,
            clippingRange=(0.2634, 0.7094),
        ) )

        # 5
        cameras.append( dict(
            pos=(-0.02598, -0.03114, -0.4640),
            focalPoint=(-8.459e-3, 5.653e-4, -5.102e-3),
            viewup=(-0.9902, 0.1367, 0.02836),
            distance=0.4604,
            clippingRange=(0.2849, 0.6823),
        ) )

        # 6
        cameras.append( dict(
            pos=(-0.1927, -0.04253, -0.4248),
            focalPoint=(-8.459e-3, 5.653e-4, -5.102e-3),
            viewup=(-0.9083, 0.1721, 0.3812),
            distance=0.4604,
            clippingRange=(0.2647, 0.7078),
        ) )

        #7
        cameras.append( dict(
            pos=(-0.08146417, -0.03583706, -0.4581959),
            focalPoint=(-8.450395e-3, 5.673127e-4, -5.082399e-3),
            viewup=(-0.9778465, 0.1505093, 0.1454758),
            distance=0.4604000,
            clippingRange=(0.2767237, 0.6955915),
        ) )

        #13
        cameras.append( dict(
            pos=(-0.4095, -0.01605, -0.2305),
            focalPoint=(-8.459e-3, 5.653e-4, -5.102e-3),
            viewup=(-0.4852, 0.2144, 0.8477),
            distance=0.4604,
            clippingRange=(0.3014, 0.6616),
        ) )

        cameras.append( dict(
            pos=(-0.4452, 0.01828, 0.1393),
            focalPoint=(-8.459e-3, 5.653e-4, -5.102e-3),
            viewup=(0.3142, 0.2221, 0.9230),
            distance=0.4604,
            clippingRange=(0.3223, 0.6352),
        ) )

        cameras.append( dict(
            pos=(-0.3468, -0.03229, 0.3054),
            focalPoint=(-8.397e-3, 5.628e-4, -5.122e-3),
            viewup=(0.6457, 0.2301, 0.7281),
            distance=0.4604,
            clippingRange=(0.2780, 0.6910),
        ) )

        # 17 en adelante
        cameras.append( dict(
            pos=(-0.1999, -0.07980, 0.4058),
            focalPoint=(-8.397e-3, 5.628e-4, -5.122e-3),
            viewup=(0.8455, 0.2870, 0.4503),
            distance=0.4604,
            clippingRange=(0.2537, 0.7216),
        ) )

        t_switch_camera = np.zeros(49)
        for i in range(8):
            t_switch_camera[i] = i
        t_switch_camera[8:13] = 7
        sw = np.linspace(13, 17, 4, dtype=np.int8)
        for i in range(len(sw)-1):
            t_switch_camera[sw[i]:sw[i+1]] = i + 8
        t_switch_camera[17:] = 11


    
    return cameras, t_switch_camera