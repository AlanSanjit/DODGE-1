# DODGE-1
DODGE-1 (Debris Observation and Dodge Guidance Engine): a Raspberry Pi 5 running the QNX RTOS acts as a satellite flight computer. A TFLite CNN and stereo cameras track a real ping-pong ball, and when it gets within 20 cm the Pi fires simulated thrusters to dodge the "asteroid" in a 3D physics simulation, with fault-tolerant process isolation.
