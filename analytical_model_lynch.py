import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def get_RotM(ang):
    sin = np.sin(ang)
    cos = np.cos(ang)
    rot_mat_z = np.matrix([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    return rot_mat_z


class PushSim():
    def __init__(self, name=None):
        """
        This is the analytical model for the push environment
        Parameters
        ---------
        name: String
            Defines the name of the block of the pendulum true model.
        """
        self.shape_height = 0.06/2
        self.shape_width = 0.20/2
        self.mu_robot_object = 0.25
        self.surface_normal = np.array([[0.0, 1.0, 0.0]])
        self.dt = 0.0667
        self.scale_l = 2.359

        # Contact-cases
        self.ang = 0.1 # robot object friction arc tan
        # Pass properties of object
        self.com_x = 0.0
        self.mass = 0.1
        self.surface_friction = 0.2

    def get_RotM(self, ang):
        sin = np.sin(ang)
        cos = np.cos(ang)
        rot_mat_z = np.matrix([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
        return rot_mat_z

    def step(self, current_state, action):

        robot_vel = action    # Defines the robot velocity, edit to make change in the action
        # print(robot_vel.shape)

        states = current_state[0:3]
        robot_pos = current_state[3:5] + robot_vel*self.dt

        rot_mat_1 = self.get_RotM(self.ang)
        rot_mat_2 = self.get_RotM(-self.ang)

        # rotate the normal to get the boundary forces
        fb_1 = np.matmul(rot_mat_1, self.surface_normal.T)
        fb_2 = np.matmul(rot_mat_2, self.surface_normal.T)
        fb_1 = np.array(fb_1.T)[0]
        fb_2 = np.array(fb_2.T)[0]

        rot_mat_object = self.get_RotM(states[2])
        state_t_com_wf = np.matmul(rot_mat_object, np.array([[self.com_x, 0, 0]]).T)
        state_t_com_wf = np.array(state_t_com_wf.T)[0][0:2] + states[0:2]

        rot_mat_robot = self.get_RotM(-states[2])
        action_t_com = np.matmul(rot_mat_robot, np.array([[robot_pos[0] - state_t_com_wf[0], robot_pos[1] - state_t_com_wf[1], 0]]).T)
        action_t_com = np.array(action_t_com.T)[0]

        contact_point_comf = np.array([action_t_com[0], action_t_com[1]])

        # Compute the bounding torques
        m_1 = contact_point_comf[0] * fb_1[1] - contact_point_comf[1] * fb_1[0]
        m_2 = contact_point_comf[0] * fb_2[1] - contact_point_comf[1] * fb_2[0]

        l = self.surface_friction * self.mass * self.scale_l
        l2 = l * l

        # Eq(6) part I
        vx_tmp1 = np.multiply(l2, fb_1[0])
        vy_tmp1 = np.multiply(l2, fb_1[1])
        vx_tmp2 = np.multiply(l2, fb_2[0])
        vy_tmp2 = np.multiply(l2, fb_2[1])

        # From 1992 Lynch, Eq 2
        vbx1 = vx_tmp1 - np.multiply(m_1, contact_point_comf[1])
        vby1 = vy_tmp1 + np.multiply(m_1, contact_point_comf[0])
        vbx2 = vx_tmp2 - np.multiply(m_2, contact_point_comf[1])
        vby2 = vy_tmp2 + np.multiply(m_2, contact_point_comf[0])

        n1 = np.sqrt(np.square(vbx1) + np.square(vby1))
        n2 = np.sqrt(np.square(vbx2) + np.square(vby2))

        action_direction_wf = np.array([[robot_vel[0], robot_vel[1], 0]])

        u_normed = np.matmul(rot_mat_robot, action_direction_wf.T)
        u_normed = np.array(u_normed.T)
        u_normed = u_normed[0]
        nu = np.sqrt(np.square(u_normed[0]) + np.square(u_normed[1]))

        # if we have the slipping case, we need to find the correct boundary velocity and the scaling factor
        cang1 = np.divide(vbx1 * u_normed[0] + vby1 * u_normed[1], n1 * nu)
        ang1 = np.rad2deg(np.arccos(cang1))
        cang2 = np.divide(vbx2 * u_normed[0] + vby2 * u_normed[1], n2 * nu)
        ang2 = np.rad2deg(np.arccos(cang2))

        # if the angle between the push and one of the boundary
        # velocities is greater than the angle between the two
        # boundary velocities, the push is sliding
        cang3 = np.divide(vbx2 * vbx1 + vby2 * vby1, n1 * n2)
        ang3 = np.arccos(cang3)

        b1 = np.array([vbx1, vby1])
        b2 = np.array([vbx2, vby2])

        if cang1 <= cang2:
            vb = b1
        else:
            vb = b2

        kappa = np.divide(self.surface_normal[0][0] * u_normed[0] + self.surface_normal[0][1] * u_normed[1],
                          np.multiply(self.surface_normal[0][0], vb[0]) + np.multiply(self.surface_normal[0][1], vb[1]))

        if cang3 < cang1 and cang3 < cang2:
            vp_out = u_normed
            # print('Sticking')
        else:
            vp_out = np.multiply(kappa, vb)
            # print('Sliding')

        ux = vp_out[0]
        uy = vp_out[1]

        rx2 = contact_point_comf[0] * contact_point_comf[0]
        ry2 = contact_point_comf[1] * contact_point_comf[1]

        div = l2 + rx2 + ry2

        tx_tmp = np.multiply((l2 + rx2), ux) + np.multiply(contact_point_comf[0],
                                                           np.multiply(contact_point_comf[1], uy))
        tx = np.divide(tx_tmp, div)

        ty_tmp = np.multiply((l2 + ry2), uy) + np.multiply(contact_point_comf[0],
                                                           np.multiply(contact_point_comf[1], ux))
        ty = np.divide(ty_tmp, div)

        rot_tmp = np.multiply(contact_point_comf[0], ty) - np.multiply(contact_point_comf[1], tx)
        rot = np.divide(rot_tmp, l2)

        ty_gc = ty - rot * self.com_x

        pos_x_wf = states[0] + (tx*np.cos(states[2]+rot*self.dt) - ty_gc*np.sin(states[2]+rot*self.dt))*self.dt
        pos_y_wf = states[1] + (tx*np.sin(states[2]+rot*self.dt) + ty_gc*np.cos(states[2]+rot*self.dt))*self.dt

        if abs(contact_point_comf[1]) > self.shape_height or abs(contact_point_comf[0]) > self.shape_width:
            return np.array([states[0], states[1], states[2]]), False, contact_point_comf, u_normed, vbx1, vby1, vbx2, vby2

        return np.array([pos_x_wf, pos_y_wf, states[2]+rot*self.dt]), True, contact_point_comf, u_normed, vbx1, vby1, vbx2, vby2


plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(10, 8))
dt = 0.0667
pos = np.array([0.0, 0.3, 0.0]) # x, y, theta
robot_pos = np.array([0.08, 0.2])

action_default = np.array([0.0, 0.05])  # vx, vy
push_model = PushSim()

# Start push simulation
for i in range(0, 200):
    current_state = np.array([pos[0], pos[1], pos[2], robot_pos[0], robot_pos[1]])
    action_to_execute = action_default
    # Update robot position with the commanded velocity - Position based velocity control
    robot_pos = robot_pos + action_to_execute * dt

    pos, made_contact, contact_point_comf, u_normed, vbx1, vby1, vbx2, vby2 = push_model.step(current_state, action_to_execute)
    ######################### Only for visualisation ######################
    ax[0].plot(pos[0], pos[1], 'xr')
    theta = pos[2]
    rot_mat_t_lc = get_RotM(theta)
    state_t_lc = np.matmul(rot_mat_t_lc, np.array([[-push_model.shape_width, -push_model.shape_height, 0]]).T)

    # Create a rectangle with the specified parameters
    rectangle = Rectangle((state_t_lc[0] + pos[0], state_t_lc[1] + pos[1]), push_model.shape_width*2, push_model.shape_height*2, angle=np.rad2deg(pos[2]), edgecolor='b', facecolor='none')

    ax[0].plot(robot_pos[0], robot_pos[1], 'Xb')

    # Add the rectangle to the axis
    ax[0].add_patch(rectangle)
    ax[0].set_xlim(-0.3, 0.3)
    ax[0].set_ylim(0.2, 1.0)

    # ax[1, 0].set_ylim(-0.2, 0.3)
    # ax[1, 1].set_ylim(-0.2, 0.3)

    ax[0].grid(True)

    # ax.plot([robot_pos[0], robot_pos[0] + action_to_execute[0]], [robot_pos[1], robot_pos[1] + action_to_execute[1]], marker='o', linestyle='-', color='red')

    rectangle_com = Rectangle((- push_model.shape_width, - push_model.shape_height), push_model.shape_width*2, push_model.shape_height*2, angle=0, edgecolor='r', facecolor='none')

    ax[1].add_patch(rectangle_com)


    # COM Frame
    ax[1].plot(contact_point_comf[0], contact_point_comf[1], 'Xb')

    ax[1].plot([contact_point_comf[0], contact_point_comf[0] + u_normed[0]], [contact_point_comf[1], contact_point_comf[1] + u_normed[1]],
        marker='o', linestyle='-')
    ax[1].plot([contact_point_comf[0], contact_point_comf[0] + 2 * vbx1], [contact_point_comf[1], contact_point_comf[1] + 2 * vby1],
        marker='.', linestyle='-',
        color='blue')
    ax[1].plot(
        [contact_point_comf[0], contact_point_comf[0] + 2 * vbx2],
        [contact_point_comf[1], contact_point_comf[1] + 2 * vby2],
        marker='.', linestyle='-',
        color='green')

    ax[0].grid(True)

    plt.pause(0.01)
    # plt.close()
    ax[1].cla()
    ax[0].cla()
