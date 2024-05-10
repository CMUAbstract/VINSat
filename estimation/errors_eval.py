import numpy as np
import ipdb
import matplotlib.pyplot as plt
        
def time_to_error():
    time_for_1 = []
    time_for_2 = []
    time_for_5 = []
    for i in range(1, 11):
        if i==4:
            continue
        errors = np.load(f'errors{i}.npy')
        times = np.load(f'times{i}.npy')
        time_for_5.append(times[np.argmax(errors<5)])
        time_for_2.append(times[np.argmax(errors<2)])
        time_for_1.append(times[np.argmax(errors<1)])
    ipdb.set_trace()

def time_to_error_hist():
    folder = "landmarks/camera_ready/dets_and_poses"
    errors = np.load(folder + '/errors.npy', allow_pickle=True)
    times = np.load(folder + '/times.npy', allow_pickle=True)
    num_trajs = len(errors)
    time_for_5 = []
    for i in range(num_trajs):
        filtered_errors = errors[i] < 5
        if np.any(filtered_errors):
            time_for_5.append(times[i][np.argmax(errors[i]<5)])
        else:
            # Handle case where error never goes below 5 km by assigning a default value or ignoring
            pass
    # ipdb.set_trace()
    time_for_5 = np.array(time_for_5)
    # Sort the times for cumulative calculation
    time_for_5_sorted = np.sort(time_for_5)
    
    # Calculate the cumulative fraction of trajectories
    # Normalize by the total number of trajectories, not just those filtered
    cumulative_fraction = np.arange(1, len(time_for_5) + 1) / num_trajs
    
    # Plotting the cumulative fraction
    plt.figure(figsize=(10, 6))
    plt.step(time_for_5_sorted, cumulative_fraction, where='post', label='Fraction of Orbits <5km Error')
    plt.title('Cumulative Fraction of First Times Reaching <5km Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Fraction of Total Orbits')
    plt.ylim(0, 1)  # Ensure y-axis goes from 0 to 1 to represent the full range of fractions
    plt.grid(True)
    plt.legend()
    plt.savefig('time_to_5.png')


if __name__ == '__main__':
    # time_to_error()
    time_to_error_hist()