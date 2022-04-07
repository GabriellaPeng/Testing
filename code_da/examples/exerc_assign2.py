import numpy as np
from datetime import datetime, timedelta

time_format = '%H:%M %p'


def charging_mechanism_detection(instance_power_data, time):
    instance_power_data, time = np.asarray(instance_power_data), np.asarray(time)
    segments = sengment_array(instance_power_data, time)

    depot_charge, oppo_charge, break_charge = [], [], []

    for seg in segments:
        if seg['charging']:
            if timedelta(minutes=3) <= seg['duration'] <= timedelta(minutes=60):
                oppo_charge.append(seg)
            elif timedelta(minutes=60) > seg['duration']:
                depot_charge.append(seg)

    charging_mechanism = {cm: {} for cm in ['regenerative_braking',
                                            'opportunity_charging',
                                            'depot_charging']
                          }

    for charging_segments in list(charging_mechanism):

        if charging_segments != 'regenerative_braking':  # handle regnerative braking
            # separately
            avg_duration, total_energy = detect_duration_n_total_energy(
                vars()[charging_segments])

            charging_mechanism[charging_segments].update(
                {'episodes': len(vars()[charging_segments]),
                 'avg_duration': avg_duration,
                 'energy_transfer': total_energy}
            )
        else:
            charging_mechanism[charging_segments].update(
                process_regenerative_braking(instance_power_data, time))


def process_regenerative_braking(instance_power_data, time):
    def _detect_repeat_element(arr):  #
        iszero = np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    repeat_elements = _detect_repeat_element(np.diff(instance_power_data))
    ind_zero_times, new_time = [], []

    for re in repeat_elements:
        if all(instance_power_data[re[0]:re[1] + 1] == 0):
            zero_range = np.arange(re[0], re[1] + 1)
            # TODO from here to improve.
            if zero_range[0] == 0 and len(time) != zero_range[-1]:
                pass
            elif zero_range[-1] != len(time):
                new_time.append(
                    param_dict(time, [ind_zero_times[i - 1][-1] + 1, zero_range[0]]))

    return {'episodes': 'continuously happening',
            'avg_duration': np.sum(new_time),
            'total_energy_transfer': np.sum(instance_power_data)}


def detect_duration_n_total_energy(segments):
    average_duration, total_energy_transfered = timedelta(minutes=0), 0

    for seg in segments:
        average_duration += seg['duration']
        total_energy_transfered += seg['total_energy']

    return average_duration / len(segments), total_energy_transfered


def param_dict(time, time_ind, squences=None, charging=None):
    duration = datetime.strptime(time[time_ind[1]], time_format) - datetime.strptime(
        time[time_ind[0]], time_format)

    if squences is None:
        return duration

    return {
        'duration': duration,
        'energy_transfer': abs(squences[-1] - squences[0]),
        'charging': charging
    }


def sengment_array(pinst, time):
    sequence, segment = [], []
    increase = None

    for i, p in enumerate(pinst):
        if i == 0:
            sequence.append(p)
            time_ini = 0
        elif p < pinst[i - 1]:
            if increase:
                segment.append(
                    param_dict(time, [time_ini, i - 1], sequence, not increase))
                time_ini = i - 1
                sequence = []
                sequence.append(pinst[i - 1])
                sequence.append(p)
            else:
                sequence.append(p)
            increase = False
        else:
            if not increase:
                segment.append(
                    param_dict(time, [time_ini, i - 1], sequence, not increase))
                time_ini = i - 1
                sequence = []
                sequence.append(pinst[i - 1])
                sequence.append(p)
            else:
                sequence.append(p)
            increase = True
    segment.append(param_dict(time, [time_ini, i], sequence, not increase))

    return segment


if __name__ == "__main__":
    # [0, -1, 1, 3, 1, -9, 7, 8, 6, 11, 22, -1, -9, 7, 8, 9, 11, -1]
    instance_power_data = [0, -1, 1, 3, 1, -9, 7, 0, 0, 0, 22, -1, -9, 7, 8, 9, 11, -1]
    time = ['1:01 AM', '1:20 AM', '2:00 AM', '3:00 AM', '4:00 AM', '5:00 AM', '6:00 AM',
            '7:00 AM', '7:10 AM', '8:10 AM', '8:11 AM', '8:14 AM', '8:15 AM', '9:11 AM',
            '10:00 AM', '11:00 AM', '11:10 AM', '11:12 AM', '12:00 AM']

    segts = charging_mechanism_detection(instance_power_data, time)
