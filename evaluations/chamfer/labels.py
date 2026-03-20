import numpy as np
import os

metal_planar_label = {
    -45: './adcData/metal/20251118_165138',
    -40: './adcData/metal/20251118_165110',
    -35: './adcData/metal/20251118_165038',
    -25: './adcData/metal/20251118_164948',
    -20: './adcData/metal/20251118_164922',
    -30: './adcData/metal/20251118_165013',
    -15: './adcData/metal/20251118_164841',
    -10: './adcData/metal/20251118_164747',
    -5: './adcData/metal/20251118_164716',
    0: './adcData/metal/20251118_163306',
    5: './adcData/metal/20251118_163350',
    10: './adcData/metal/20251118_163450',
    15: './adcData/metal/20251118_163839',
    20: './adcData/metal/20251118_163925',
    25: './adcData/metal/20251118_163950',
    30: './adcData/metal/20251118_164018',
    35: './adcData/metal/20251118_164042',
    40: './adcData/metal/20251118_164108',
    45: './adcData/metal/20251118_164134',
    'length': 0.76,
    'loc': [-0.13,1.41],
    'name': 'Metal',
}

plastic_planar_label = {
    -45: './adcData/plastic/20251119_153135',
    -40: './adcData/plastic/20251119_153220',
    -35: './adcData/plastic/20251119_153319',  # Used last entry (2 total found)
    -30: './adcData/plastic/20251119_153348',
    -25: './adcData/plastic/20251119_153417',
    -20: './adcData/plastic/20251119_153454',  # Used last entry (2 total found)
    -15: './adcData/plastic/20251119_153525',
    -10: './adcData/plastic/20251119_153558',
    -5:  './adcData/plastic/20251119_153638',
    0:   './adcData/plastic/20251119_153707',
    5:   './adcData/plastic/20251119_153740',  # Used last entry (2 total found)
    10:  './adcData/plastic/20251119_154022',  # Used last entry (4 total found)
    15:  './adcData/plastic/20251119_154059',
    20:  './adcData/plastic/20251119_154130',
    25:  './adcData/plastic/20251119_154202',
    30:  './adcData/plastic/20251119_154238',
    35:  './adcData/plastic/20251119_154309',
    40:  './adcData/plastic/20251119_154339',
    45:  './adcData/plastic/20251119_154416',
    'length': 0.90,
    'loc': [-0.13,1.41],
    'name': 'Plastic',
}


fabrics_planar_label = {
    -45: './adcData/fabrics/20251118_173901',
    -40: './adcData/fabrics/20251118_174225',
    -35: './adcData/fabrics/20251118_174256',
    -30: './adcData/fabrics/20251118_174414',
    -25: './adcData/fabrics/20251118_174452',
    -20: './adcData/fabrics/20251118_174719',
    -15: './adcData/fabrics/20251118_174757',
    -10: './adcData/fabrics/20251118_174832',
    -5:  './adcData/fabrics/20251118_174908',
    0:   './adcData/fabrics/20251118_174948',
    5:   './adcData/fabrics/20251118_175337',  # Used last entry (2 total found)
    10:  './adcData/fabrics/20251118_175409',  # Used last entry (2 total found)
    15:  './adcData/fabrics/20251118_175437',
    20:  './adcData/fabrics/20251118_175504',
    25:  './adcData/fabrics/20251118_175544',
    30:  './adcData/fabrics/20251118_175616',
    35:  './adcData/fabrics/20251118_175645',
    40:  './adcData/fabrics/20251118_175718',
    45:  './adcData/fabrics/20251118_175745',
    'length': 0.76,
    'loc': [-0.13,1.41],
    'name': 'Fabrics',
}


drywall_planar_label = {
    -45: './adcData/drywall/20251119_143242',
    -40: './adcData/drywall/20251119_143325',
    -35: './adcData/drywall/20251119_143402',
    -30: './adcData/drywall/20251119_143704',
    -25: './adcData/drywall/20251119_143747',
    -20: './adcData/drywall/20251119_143906',
    -15: './adcData/drywall/20251119_144028',
    -10: './adcData/drywall/20251119_144247',
    -5:  './adcData/drywall/20251119_144355',
    0:   './adcData/drywall/20251119_144513',
    5:   './adcData/drywall/20251119_144606',
    10:  './adcData/drywall/20251119_144639',
    15:  './adcData/drywall/20251119_145004',
    20:  './adcData/drywall/20251119_145849',  # Used last entry (4 total found)
    30:  './adcData/drywall/20251119_145927',
    35:  './adcData/drywall/20251119_145959',
    40:  './adcData/drywall/20251119_150033',
    45:  './adcData/drywall/20251119_150107',
    'length': 0.85,
    'loc': [-0.13,1.41],
    'name': 'Drywall',
}


wood_planar_label = {
    -45: './adcData/wood/20251119_150815',
    -40: './adcData/wood/20251119_150910',
    -35: './adcData/wood/20251119_151018',
    -30: './adcData/wood/20251119_151059',
    -25: './adcData/wood/20251119_151341',  # Used last entry (2 total found)
    -20: './adcData/wood/20251119_151415',
    -15: './adcData/wood/20251119_151442',
    -10: './adcData/wood/20251119_151511',
    -5:  './adcData/wood/20251119_151623',
    0:   './adcData/wood/20251119_151657',
    5:   './adcData/wood/20251119_151727',
    10:  './adcData/wood/20251119_151755',
    15:  './adcData/wood/20251119_151821',
    20:  './adcData/wood/20251119_151903',
    25:  './adcData/wood/20251119_151930',
    30:  './adcData/wood/20251119_152057',
    35:  './adcData/wood/20251119_152125',
    40:  './adcData/wood/20251119_152156',
    45:  './adcData/wood/20251119_152225',
    'length': 0.865,
    'loc': [-0.13,1.41],
    'name': 'Wood',
}


curvature_label = {
    -5: './adcData/curve/20251119_162643',
    -4: './adcData/curve/20251119_164600',
    -3: './adcData/curve/20251119_164714',
    -2: './adcData/curve/20251119_164839',
    -1: './adcData/curve/20251119_165041',
    # 0:  './adcData/curve/20251119_155857',
    1:  './adcData/curve/20251119_161104',
    2:  './adcData/curve/20251119_161717',
    3:  './adcData/curve/20251119_161926',
    4:  './adcData/curve/20251119_162113',
    5:  './adcData/curve/20251119_162420',    # Used last entry (2 total found)
    'loc_positive_curv': [-0.13, 0.93],
    'loc_negative_curv': [-0.13, 1.12],
    'length': 0.75,
    'name': 'Curvature',
}



def planar_gt_pc(angle_deg, loc, length, n_points=500):
    theta = np.deg2rad(angle_deg)  # Convert angle to radians
    # Linear points from -L/2 to +L/2 along the segment
    ts = np.linspace(-0.5, 0.5, n_points) * length
    pc_x = loc[0] + ts * np.cos(theta)
    pc_y = loc[1] + ts * np.sin(theta)
    pc = np.stack([pc_x, pc_y], axis=-1)
    return pc


def curvature_gt_pc(curvature, n_points=500):
    if curvature >= 0:
        loc = curvature_label['loc_positive_curv']
    else:
        loc = curvature_label['loc_negative_curv']
        
    length = curvature_label['length']

    # Handle zero curvature as a straight horizontal segment centered at loc
    if curvature == 0:
        ts = np.linspace(-0.5, 0.5, n_points) * length
        pc_x = loc[0] + ts
        pc_y = np.full_like(pc_x, loc[1])
        return np.stack([pc_x, pc_y], axis=-1)

    # Geometric radius and sign
    R = - 1.0 / np.abs(curvature)
    alpha = length / R  # subtended angle in radians

    # Circle center and central angle depend on curvature sign
    if curvature > 0:
        # Arc bulges upward; center below loc; apex at theta0 = pi/2
        Cx, Cy = loc[0], loc[1] - R
        theta0 = np.pi / 2.0
    else:
        # Arc bulges downward; center above loc; apex at theta0 = -pi/2
        Cx, Cy = loc[0], loc[1] + R
        theta0 = -np.pi / 2.0

    thetas = theta0 + np.linspace(-alpha/2.0, alpha/2.0, n_points)
    pc_x = Cx + R * np.cos(thetas)
    pc_y = Cy + R * np.sin(thetas)
    pc = np.stack([pc_x, pc_y], axis=-1)

    return pc



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    gx = np.linspace(-0.834, 0.72, 128)
    gy = np.linspace(-0.2+0.2, 2-0.3, 128)
    # pc_gt = planar_gt_pc(0, metal_planar_label)
    # print(pc_gt)

    pc_gt = curvature_gt_pc(-2)
    print(pc_gt)

    plt.figure()
    plt.scatter(pc_gt[:,0], pc_gt[:,1], s=2, c='b')
    plt.scatter(curvature_label['loc_positive_curv'][0], curvature_label['loc_positive_curv'][1], s=2, c='r')
    plt.scatter(curvature_label['loc_negative_curv'][0], curvature_label['loc_negative_curv'][1], s=2, c='g')
    plt.xlim(gx[0], gx[-1])
    plt.ylim(gy[0], gy[-1])
    # Set equal aspect ratio according to data extent
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Ground Truth Point Cloud')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.show()





