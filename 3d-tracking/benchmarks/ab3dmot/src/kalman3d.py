"""
Linear Kalman filter for 3D bounding box state estimation.

State (9-dim):  [x, y, z, l, w, h, yaw, vx, vy]
Observation (7-dim): [x, y, z, l, w, h, yaw]

Constant-velocity model in XY; dimensions and yaw treated as near-constant.
No vertical velocity — pedestrians are largely ground-plane objects.
"""
from __future__ import annotations

import numpy as np


class KalmanFilter3D:
    DIM_X = 9
    DIM_Z = 7

    def __init__(self, initial_box: np.ndarray, dt: float = 1.0) -> None:
        """
        initial_box: [x, y, z, l, w, h, yaw]
        dt: time step between frames (default 1 frame)
        """
        self.dt = float(dt)

        # State vector
        self.x = np.zeros(self.DIM_X, dtype=np.float64)
        self.x[:7] = initial_box[:7]

        # State transition F: x += vx*dt, y += vy*dt, rest constant
        self.F = np.eye(self.DIM_X, dtype=np.float64)
        self.F[0, 7] = self.dt
        self.F[1, 8] = self.dt

        # Measurement matrix H: observe the first 7 state dims
        self.H = np.zeros((self.DIM_Z, self.DIM_X), dtype=np.float64)
        self.H[:7, :7] = np.eye(self.DIM_Z, dtype=np.float64)

        # Process noise Q
        self.Q = np.diag([
            1.0, 1.0, 1.0,    # position uncertainty
            1.0, 1.0, 1.0,    # dimension uncertainty
            0.5,              # yaw uncertainty
            0.25, 0.25,       # velocity uncertainty
        ]).astype(np.float64)

        # Measurement noise R
        self.R = np.diag([
            0.5, 0.5, 1.0,    # position measurement noise
            1.0, 1.0, 1.0,    # dimension measurement noise
            0.5,              # yaw measurement noise
        ]).astype(np.float64)

        # State covariance P — high initial velocity uncertainty
        self.P = np.diag([
            10.0, 10.0, 10.0,
            10.0, 10.0, 10.0,
            10.0,
            100.0, 100.0,
        ]).astype(np.float64)

    def predict(self) -> np.ndarray:
        """Predict next state; returns predicted box [x, y, z, l, w, h, yaw]."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:7].copy()

    def update(self, observation: np.ndarray) -> None:
        """Update state with measurement [x, y, z, l, w, h, yaw]."""
        z = np.asarray(observation[:7], dtype=np.float64)
        y = z - self.H @ self.x

        # Normalise yaw innovation to [-π, π]
        while y[6] > np.pi:
            y[6] -= 2.0 * np.pi
        while y[6] < -np.pi:
            y[6] += 2.0 * np.pi

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.DIM_X, dtype=np.float64) - K @ self.H) @ self.P

    @property
    def state(self) -> np.ndarray:
        return self.x[:7].copy()
