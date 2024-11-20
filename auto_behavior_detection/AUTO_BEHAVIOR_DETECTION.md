# **Automated** Behavior Detection

Data Preprocess
---------------

1. Project each waypoint onto the target trajectory centerline, then calculate the position of the projected point and record these data.
2. Apply the *butter_lowpass_filter()* function to filter the speed of all vehicles.
3. Compute the acceleration of each vehicle.
4. Pair the projected positions of the ego car and NPC cars in each frame, and invoke the *get_action()* function to analyze the state between the ego car and the target NPC car (EGO_BEFORE_NPC = 1, EGO_NEAR_NPC = 0, EGO_AFTER_NPC = -1).
5. Group frames with consecutive states, and based on the positional relationship of the states within each group (before -> after, after -> before, before -> near, after -> near), identify the starting and ending frames of the behavior.


## Rule-based Behavior Detection

Different interactions involve different groups, and we can judge them by basic rules:

1. No-Interaction: the average distance between two vehicles exceeds 30m
2. Bypass: the speed of a NPC car is less than $0.1 m/s$
3. Yield: the speed of the ego car is less than the speed of a NPC car
4. Yield: the minimum acceleration of the ego car is less than $-10 m/s^2$
5. Overtake: the acceleration of the ego car is greater than $2 m/s^2$
