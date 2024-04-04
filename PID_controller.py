import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd, dt):
        """
        Initialize the PID controller with gains and time step.

        @param kp: Proportional gain, a d_control size vector.
        @param ki: Integral gain, a d_control size vector.
        @param kd: Derivative gain, a d_control size vector.
        @param dt: Time step (constant time difference between updates).
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt  # Time step

        self.integral = np.zeros_like(kp, dtype=np.float64)
        self.last_error = np.zeros_like(kp, dtype=np.float64)
        self.end_point = np.zeros_like(kp, dtype=np.float64)

    def reset_endpoint(self, end_point):
        assert len(end_point) == len(self.end_point), "Endpoint dimension mismatch"
        self.end_point = end_point
        self.integral = np.zeros_like(self.kp, dtype=np.float64)
        self.last_error = np.zeros_like(self.kp, dtype=np.float64)

    def control(self, current_state):
        """
        Calculate control signal based on the current state.

        @param current_state: The current state of the system
        """
        error = self.end_point - current_state
        print(np.round(error, 4))

        # Update the integral and derivative terms considering the constant time difference
        self.integral += error * self.dt
        derivative = (error - self.last_error) / self.dt if self.dt != 0 else 0.0
        self.last_error = error

        # Compute the control output
        control = self.kp * error + self.ki * self.integral + self.kd * derivative

        return control