from . import BaseController
import numpy as np

class Controller(BaseController):
  def __init__(self):
    self.p = 0.3
    self.i = 0.15
    self.d = -0.1
    self.error_integral = 0
    self.prev_error = 0
    self.max_integral = 2.0
    self.prev_output = 0
    self.alpha = 0.5

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = target_lataccel - current_lataccel
    
    # Update integral term with clamping
    self.error_integral += error
    self.error_integral = np.clip(self.error_integral, -self.max_integral, self.max_integral)
    
    # Calculate derivative term
    error_diff = error - self.prev_error
    self.prev_error = error
    
    # Compute PID output
    output = self.p * error + self.i * self.error_integral + self.d * error_diff
    
    # Apply smoothing with previous output
    output = self.alpha * output + (1 - self.alpha) * self.prev_output
    self.prev_output = output
    
    return output
