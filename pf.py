from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from . pf_base import PFLocaliserBase
from . util import rotateQuaternion, getHeading
import math
import random


class PFLocaliser(PFLocaliserBase):

    def __init__(self, logger, clock):
        # ----- Call the superclass constructor
        super().__init__(logger, clock)

        # ----- Set motion model parameters

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict


    def initialise_particle_cloud(self, initialpose):

        particles = PoseArray()

        initial_x = initialpose.pose.pose.position.x
        initial_y = initialpose.pose.pose.position.y
        initial_theta = getHeading(initialpose.pose.pose.orientation)

        total_number_of_particles = 400
        sigma_x = 12.0
        sigma_y = 12.0
        sigma_theta = 0.5

        for i in range(total_number_of_particles):
            particle_x = random.gauss(initial_x, sigma_x)
            particle_y = random.gauss(initial_y, sigma_y)
            particle_theta = random.gauss(initial_theta, sigma_theta)

            pose = Pose()
            pose.position = Point(x = particle_x, y = particle_y, z = 0.0)
            pose.orientation = rotateQuaternion(Quaternion(w = 1.0), particle_theta)

            particles.poses.append(pose)

        self.particlecloud = particles
        return particles


        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.

        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """

    def update_particle_cloud(self, scan):

        particles = self.particlecloud.poses
        number_of_particle = len(particles)

        if number_of_particle == 0 or scan is None:
            return

        weights = []
        for p in particles:
            weights.append(self.sensor_model.get_weight(scan, p))

        sum_of_weights = sum(weights)
        if sum_of_weights <= 0.0:
            return

        normalized_weights  = []
        for weight in weights:
            normalized_weights.append(weight/sum_of_weights)

        weights = normalized_weights

        new_particles = []
        random_offset = random.uniform(0.0, 1.0 / number_of_particle)
        cumulative_weight = weights[0]
        weight_index = 0

        resampling_sigma_x = 0.15
        resampling_sigma_y = 0.15
        resampling_sigma_theta = 0.5

        for m in range(number_of_particle):
            cdf_position = random_offset + (m/number_of_particle)
            while cdf_position > cumulative_weight and weight_index < number_of_particle - 1:
                weight_index += 1
                cumulative_weight += weights[weight_index]

            selected_particle = particles[weight_index]
            pose = Pose()

            pose.position.x = selected_particle.position.x + random.gauss(0.0, resampling_sigma_x)
            pose.position.y = selected_particle.position.y + random.gauss(0.0, resampling_sigma_y)
            pose.position.z = selected_particle.position.z
            particle_theta = getHeading(selected_particle.orientation) + random.gauss(0.0, resampling_sigma_theta)
            pose.orientation = rotateQuaternion(Quaternion(w = 1.0), particle_theta)

            new_particles.append(pose)

        self.particlecloud.poses = new_particles
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.

        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """

    def estimate_pose(self):

        particles = self.particlecloud.poses
        number_of_particle = len(particles)

        if number_of_particle == 0:
            print("Error")
            return Pose()

        xs = []
        ys = []

        for pose in particles:
            xs.append(pose.position.x)
            ys.append(pose.position.y)

        mean_x = sum(xs) / number_of_particle
        mean_y = sum(ys) / number_of_particle

        distances = []
        for index, particle in enumerate(particles):
            dx = particle.position.x - mean_x
            dy = particle.position.y - mean_y
            distances.append((pow(dx,2) + pow(dy,2),index))

        def get_distance(item):
            return item[0]

        distances.sort(key=get_distance)
        how_many_keep = number_of_particle // 2

        keep_indices = []
        for i in range(how_many_keep):
            distance, index = distances[i]
            keep_indices.append(index)

        number_of_keep_particles = len(keep_indices)

        sum_x = 0.0
        sum_y = 0.0

        for i in keep_indices:
            sum_x += particles[i].position.x
            sum_y += particles[i].position.y

        estimate_x = sum_x / number_of_keep_particles
        estimate_y = sum_y / number_of_keep_particles

        sum_quaternion_x = 0.0
        sum_quaternion_y = 0.0
        sum_quaternion_z = 0.0
        sum_quaternion_w = 0.0

        for i in keep_indices:
            sum_quaternion_x += particles[i].orientation.x
            sum_quaternion_y += particles[i].orientation.y
            sum_quaternion_z += particles[i].orientation.z
            sum_quaternion_w += particles[i].orientation.w

        quaternion_x = sum_quaternion_x / number_of_keep_particles
        quaternion_y = sum_quaternion_y / number_of_keep_particles
        quaternion_z = sum_quaternion_z / number_of_keep_particles
        quaternion_w = sum_quaternion_w / number_of_keep_particles

        normalized_quaternion = math.sqrt(pow(quaternion_x,2) + pow(quaternion_y,2) + pow(quaternion_z,2)+ pow(quaternion_w,2))

        if normalized_quaternion > 1e-12:
            quaternion_x /= normalized_quaternion
            quaternion_y /= normalized_quaternion
            quaternion_z /= normalized_quaternion
            quaternion_w /= normalized_quaternion
        else:
            quaternion_x = 0.0
            quaternion_y = 0.0
            quaternion_z = 0.0
            quaternion_w = 1.0

        estimate_particle_pose = Pose()
        estimate_particle_pose.position.x = estimate_x
        estimate_particle_pose.position.y = estimate_y
        estimate_particle_pose.position.z = 0.0
        estimate_particle_pose.orientation.x = quaternion_x
        estimate_particle_pose.orientation.y = quaternion_y
        estimate_particle_pose.orientation.z = quaternion_z
        estimate_particle_pose.orientation.w = quaternion_w


        return estimate_particle_pose
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).

        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.

        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """
