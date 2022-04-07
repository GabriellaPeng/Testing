import numpy as np
from datetime import datetime, timedelta


# assume
# if cumulative instantaneous power(instance_power_data) is continuously decreasing (i.e.
# charging):
# > 1 hour -- depot charging
# [3min, 60min] -- opportunity charging
# < 3mins -- Regenerative braking


def charging_mechanism_detection(instance_power_data, time):
    '''
    main function: Extracting the number of episodes, the average episode duration,
    and the total energy transferred for each of the three battery charging mechanisms
    :param instance_power_data | list of float of cumulative instantaneous power kwh [-200, -198, ...]
    :param time | same length as instance_power_data, list of string of datetime ['1:01 AM', '1:20 AM', '2:00 AM', '3:00 AM']
    :return: {}
    '''
    instance_power_data, time = np.asarray(instance_power_data), np.asarray(time)
    segments = power_curve_segmentation(instance_power_data, time)

    regenerative_braking = []
    opportunity_charging = []
    depot_charging = []
    for seg in segments:
        if not seg['up']:  # only decreased segments used three charging mechanisms
            if seg['duration'] > timedelta(minutes=60):
                # detect regenerative braking mechanism
                depot_charging.append(seg)
            elif timedelta(minutes=3) <= seg['duration'] <= timedelta(minutes=60):
                # detect Opportunity charging
                opportunity_charging.append(seg)
            else:
                # detect Depot charging
                regenerative_braking.append(seg)

    charging_mechanism = {'regenerative_braking': {},
                          'opportunity_charging': {},
                          'depot_charging': {}
                          }

    for charging_segments in ['regenerative_braking', 'opportunity_charging',
                              'depot_charging']:
        avg_duration, total_energy = detect_duration_n_total_energy(
            vars()[charging_segments])
        charging_mechanism[charging_segments].update(
            {'episodes': len(vars()[charging_segments]),
             'avg_duration': avg_duration,
             'total_energy': total_energy}
        )
    return charging_mechanism


def detect_duration_n_total_energy(segments):
    average_duration, total_energy_transfered = timedelta(minutes=0), 0

    for seg in segments:
        average_duration += seg['duration']
        total_energy_transfered += seg['total_energy']

    return average_duration / len(segments), total_energy_transfered


def power_curve_segmentation(instance_power_data, time):
    '''
    split instance_power_data into several one-way increasing or decreasing
    segments. For each segment, the corresponding time duration, energy consumption have
    recorded.
    :return: list of dictionary | each dictionary indicating one monotone increasing,
    or monotone decreasing segment
                    {
                        'start_time':
                        'end_time':
                        'up': True or False | True indicating an increasing segment
                        'duration':
                        'total_energy':
                        'sequence':
                    }
    '''
    segments, sequence = [], [ ]
    up = None

    for i, p in enumerate(instance_power_data):
        if i == 0:
            start_index = 0
            sequence.append(p)
        else:
            if instance_power_data[i - 1] > p:
                if up is True:
                    segments.append({
                        'start_time': datetime.strptime(time[start_index], '%I:%M %p'),
                                                            # '1900-01-01  1:33PM'
                        'end_time': datetime.strptime(time[i - 1], '%I:%M %p'),
                        'up': up,
                        'duration': datetime.strptime(time[i - 1], '%I:%M %p')
                                    - datetime.strptime(time[start_index], '%I:%M %p'),
                        'total_energy': sequence[-1] - sequence[0],
                        'sequence': sequence
                    })
                    sequence = []
                    sequence.append(instance_power_data[i - 1])
                    sequence.append(p)
                    start_index = i - 1
                else:
                    sequence.append(p)
                up = False
            else:
                if up is False:
                    segments.append({
                        'start_time': datetime.strptime(time[start_index], '%I:%M %p'),
                        'end_time': datetime.strptime(time[i - 1], '%I:%M %p'),
                        'up': up,
                        'duration': datetime.strptime(time[i - 1], '%I:%M %p')
                                    - datetime.strptime(time[start_index], '%I:%M %p'),
                        'total_energy': sequence[-1] - sequence[0],
                        'sequence': sequence
                    })
                    sequence = []
                    sequence.append(instance_power_data[i - 1])
                    sequence.append(p)
                    start_index = i - 1
                else:
                    sequence.append(p)
                up = True

    segments.append({
        'start_time': datetime.strptime(time[start_index], '%I:%M %p'),
        'end_time': datetime.strptime(time[i], '%I:%M %p'),
        'up': up,
        'duration': datetime.strptime(time[i], '%I:%M %p')
                    - datetime.strptime(time[start_index], '%I:%M %p'),
        'total_energy': sequence[-1] - sequence[0],
        'sequence': sequence
    })
    return segments


if __name__ == "__main__":
    instance_power_data = [0, -1, 1, 3, 1, -9, 7, 8, 6, 11, 22, -1, -9, 7, 8, 9, 11, -1]
    time = ['1:01 AM', '1:20 AM', '2:00 AM', '3:00 AM', '4:00 AM', '5:00 AM', '6:00 AM',
            '7:00 AM', '7:10 AM', '8:10 AM', '8:11 AM', '8:14 AM', '8:15 AM', '9:11 AM',
            '10:00 AM', '11:00 AM', '11:10 AM', '11:12 AM', '12:00 AM']

    segts = charging_mechanism_detection(instance_power_data, time)
